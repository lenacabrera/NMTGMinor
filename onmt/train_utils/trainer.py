from __future__ import division

import onmt
import onmt.markdown
import onmt.modules
import torch
from torch.autograd import Variable
import math
import time, datetime
import os
import re
from onmt.model_factory import init_model_parameters
from onmt.utils import checkpoint_paths, normalize_gradients
from apex import amp
from onmt.train_utils.stats import Logger
from onmt.model_factory import build_model, build_language_model, optimize_model
import sys

class BaseTrainer(object):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt):

        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.dicts = dicts
        self.opt = opt
        self.cuda = (len(opt.gpus) >= 1 and opt.gpus[0] >= 0)

        self.loss_function = loss_function
        self.start_time = 0

        self.additional_data = []

    def add_additional_data(self, d, ratio):
        self.additional_data = d
        if ratio == "-1":
            self.additional_data_ratio = [1] * (len(self.additional_data + 1))
        else:
            self.additional_data_ratio = [int(s) for s in ratio.split(";")]
            assert (len(self.additional_data_ratio) == len(self.additional_data) + 1)

    def run(self, *args, **kwargs):

        raise NotImplementedError

    def eval(self, data):

        raise NotImplementedError

    def load_encoder_weight(self, checkpoint_file):

        print("Loading pretrained models from %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)

        pretrained_model = build_model(checkpoint['opt'], checkpoint['dicts'])
        pretrained_model.load_state_dict(checkpoint['model'])

        print("Loading pretrained encoder weights ...")
        pretrained_model.encoder.language_embedding = None
        enc_language_embedding = self.model.encoder.language_embedding
        self.model.encoder.language_embedding = None
        encoder_state_dict = pretrained_model.encoder.state_dict()

        self.model.encoder.load_state_dict(encoder_state_dict)
        self.model.encoder.language_embedding = enc_language_embedding
        return

    def _get_grads(self):
        grads = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                raise RuntimeError('Model parameter did not receive gradient: ' + name + '. '
                                                                                         'Use the param in the forward pass or set requires_grad=False.' +
                                   ' If you are using Stochastic model + fp16 - try to increase the number of minibatches' +
                                   ' each update to avoid uninitialized gradients.')
            grads.append(p.grad.data)
        return grads

    def _get_flat_grads(self, out=None):
        grads = self._get_grads()
        if out is None:
            grads_size = sum(g.numel() for g in grads)
            out = grads[0].new(grads_size).zero_()
        offset = 0
        for g in grads:
            numel = g.numel()
            out[offset:offset + numel].copy_(g.view(-1))
            offset += numel
        return out[:offset]


class XETrainer(BaseTrainer):

    def __init__(self, model, loss_function, train_data, valid_data, dicts, opt, setup_optimizer=True):
        super().__init__(model, loss_function, train_data, valid_data, dicts, opt)

        if self.cuda:
            torch.cuda.set_device(self.opt.gpus[0])
            torch.manual_seed(self.opt.seed)
            self.loss_function = self.loss_function.cuda()
            self.model = self.model.cuda()

        if setup_optimizer:

            self.optim = onmt.Optim(opt)
            self.optim.set_parameters(self.model.parameters())

            if not self.opt.fp16:
                opt_level = "O0"
                keep_batchnorm_fp32 = False
            elif self.opt.fp16_mixed:
                opt_level = "O1"
                keep_batchnorm_fp32 = None
            else:
                opt_level = "O2"
                keep_batchnorm_fp32 = False

            if self.cuda:
                self.model, self.optim.optimizer = amp.initialize(self.model,
                                                                  self.optim.optimizer,
                                                                  opt_level=opt_level,
                                                                  keep_batchnorm_fp32=keep_batchnorm_fp32,
                                                                  loss_scale="dynamic",
                                                                  verbosity=1)
        # An ugly hack to switch between align right and align left
        if hasattr(self.model, 'relative'):
            if self.model.relative:
                self.train_data.src_align_right = True
                self.train_data.tgt_align_right = False
                self.valid_data.src_align_right = True
                self.valid_data.tgt_align_right = False

    def save(self, epoch, valid_ppl, batch_order=None, iteration=-1):

        opt = self.opt
        model = self.model
        dicts = self.dicts

        model_state_dict = self.model.state_dict()
        optim_state_dict = self.optim.state_dict()

        #  drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dicts,
            'opt': opt,
            'epoch': epoch,
            'iteration': iteration,
            'batch_order': batch_order,
            'optim': optim_state_dict,
            'additional_batch_order': getattr(self, 'additional_batch_order', None),
            'additional_data_iteration': getattr(self, 'additional_data_iteration', None),
            'amp': amp.state_dict()
        }

        file_name = '%s_ppl_%.6f_e%.2f.pt' % (opt.save_model, valid_ppl, epoch)
        print('Writing to %s' % file_name)
        torch.save(checkpoint, file_name)

        # check the save directory here
        checkpoint_dir = os.path.dirname(opt.save_model)
        existed_save_files = checkpoint_paths(checkpoint_dir)
        for save_file in existed_save_files[opt.keep_save_files:]:
            print(" * Deleting old save file %s ...." % save_file)
            os.remove(save_file)

        best_epoch = float(re.search("_e(.*)\.pt", existed_save_files[0]).group(1))

        if epoch - best_epoch >= opt.early_stop_if_no_change:
            print(" * Early stopping at epoch %s as best epoch was %s ." % (epoch, best_epoch))
            sys.exit(0)

    def eval(self, data):
        total_loss = 0
        total_words = 0
        total_adv_loss = 0.0
        total_src_words = 0.0
        total_predict, correct_predict = 0.0, 0.0
        opt = self.opt

        # batch_order = data.create_order(random=False)
        self.model.eval()
        self.model.reset_states()

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        """ PyTorch semantics: save space by not creating gradients """
        with torch.no_grad():
            for i in range(len(data)):

                batch = data.next()[0]

                # if opt.streaming:
                #     if data.is_new_stream():
                #         streaming_state = self.model.init_stream()
                # else:
                #     streaming_state = None

                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                """ outputs can be either 
                        hidden states from decoder or
                        prob distribution from decoder generator
                """
                targets = batch.get('target_output')
                tgt_mask = targets.ne(onmt.constants.PAD)
                outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                     mirror=opt.mirror_loss, streaming_state=streaming_state)

                if opt.streaming:
                    streaming_state = outputs['streaming_state']

                outputs['tgt_mask'] = tgt_mask

                # normal loss
                loss_dict = self.loss_function(outputs, targets, model=self.model)
                loss_data = loss_dict['data']

                total_loss += loss_data
                total_words += batch.tgt_size

                total_src_words += batch.src_size
                # adv loss
                if False:
                    targets_src_lang = batch.get('targets_source_lang')
                    classifier_loss_dict = self.loss_function(outputs, targets=targets_src_lang, model=self.model,
                                                              lan_classifier=True,
                                                              reverse_landscape=opt.reverse_loss_landscape)
                    classifier_loss_data = classifier_loss_dict['data'] if classifier_loss_dict['data'] is not None else 0

                    total_adv_loss += classifier_loss_data


                # TODO: get prediction accuracy here!
                # logprobs_lan = outputs['logprobs_lan']
                # pad
                # logprobs_lan = logprobs_lan.masked_fill(outputs['src_mask'].permute(2, 0, 1), 0).type_as(logprobs_lan)
                # pred = torch.exp(logprobs_lan)

                # res = torch.tensor([0.0, 0.0, 0.0])

                # for sent_idx in range(pred.shape[1]):  # T, B, V, for each sentence
                #     for pos_idx in range(pred.shape[0]):  # for each token
                #         if not torch.all(pred[pos_idx, sent_idx, :] == 1.0):  # if not at padded positions.
                #
                #             if torch.abs(1.0 - torch.sum(pred[pos_idx, sent_idx, :])) < 0.001:
                #                 pred_idx = torch.argmax(pred[pos_idx, sent_idx, :], dim=0)
                #
                #                 total_predict += 1.0
                #                 if pred_idx == targets_src_lang[pos_idx, sent_idx]:
                #                     correct_predict += 1
                #                 # print('predicted', pred_idx, 'target', targets_src_lang[pos_idx, sent_idx])
                #                 res[pred_idx] += 1.0
                #             else:
                #                 print(pred[pos_idx, sent_idx, :])

                # print(res / torch.sum(res))

        # print('Accuracy', correct_predict / total_predict)
                # print('total_adv_loss', total_adv_loss, 'total_src_words', total_src_words)
        self.model.train()
        return total_loss / total_words, total_adv_loss / total_src_words

    def train_epoch(self, epoch, resume=False, batch_order=None, iteration=0):

        opt = self.opt
        train_data = self.train_data
        streaming = opt.streaming

        # Clear the gradients of the model
        # self.runner.zero_grad()
        self.model.zero_grad()
        self.model.reset_states()

        if resume:
            train_data.batch_order = batch_order
            train_data.set_index(iteration)
            print("Resuming from iteration: %d" % iteration)
        else:
            batch_order = train_data.create_order()
            iteration = 0

        total_tokens, total_loss, total_words = 0, 0, 0
        total_non_pads = 0
        report_loss, report_tgt_words = 0, 0
        report_classifier_loss = 0.000001
        report_src_words = 0
        start = time.time()
        n_samples = len(train_data)

        counter = 0
        num_accumulated_words = 0
        num_accumulated_sents = 0
        denom = 3584
        nan = False

        if opt.streaming:
            streaming_state = self.model.init_stream()
        else:
            streaming_state = None

        for i in range(iteration, n_samples):

            curriculum = (epoch < opt.curriculum)

            batches = [train_data.next(curriculum=curriculum)[0]]

            if (len(self.additional_data) > 0 and
                    i % self.additional_data_ratio[0] == 0):
                for j in range(len(self.additional_data)):
                    for k in range(self.additional_data_ratio[j + 1]):
                        if self.additional_data_iteration[j] == len(self.additional_data[j]):
                            self.additional_data_iteration[j] = 0
                            self.additional_data[j].shuffle()
                            self.additional_batch_order[j] = self.additional_data[j].create_order()

                        batches.append(self.additional_data[j].next()[0])
                        self.additional_data_iteration[j] += 1

            for b in range(len(batches)):
                batch = batches[b]
                if self.cuda:
                    batch.cuda(fp16=self.opt.fp16 and not self.opt.fp16_mixed)

                # if opt.streaming:
                #     if train_data.is_new_stream():
                #         streaming_state = self.model.init_stream()
                # else:
                #     streaming_state = None

                oom = False
                try:
                    # outputs is a dictionary containing keys/values necessary for loss function
                    # can be flexibly controlled within models for easier extensibility
                    targets = batch.get('target_output')
                    tgt_mask = targets.data.ne(onmt.constants.PAD)

                    outputs = self.model(batch, streaming=opt.streaming, target_mask=tgt_mask,
                                         zero_encoder=opt.zero_encoder,
                                         mirror=opt.mirror_loss, streaming_state=streaming_state)

                    batch_size = batch.size
                    #
                    outputs['tgt_mask'] = tgt_mask
                    #
                    loss_dict = self.loss_function(outputs, targets, model=self.model)
                    loss_data = loss_dict['data']
                    loss = loss_dict['loss'].div(denom)  # a little trick to avoid gradient overflow with fp16

                    optimizer = self.optim.optimizer

                    has_classifier_loss = self.opt.language_classifier and (not self.opt.freeze_language_classifier)

                    # don't even do backprop if both encoder and decoder are frozen
                    if not (self.opt.freeze_encoder and self.opt.freeze_decoder):
                        if self.cuda:
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                scaled_loss.backward(retain_graph=has_classifier_loss)
                        else:
                            loss.backward(retain_graph=has_classifier_loss)

                    # alternate to language clasifier loss
                    if self.opt.language_classifier:
                        # src_lang = batch.get('source_lang')
                        targets_src_lang = batch.get('targets_source_lang')
                        classifier_loss_dict = self.loss_function(outputs, targets=targets_src_lang, model=self.model,
                                                       lan_classifier=True, reverse_landscape=self.opt.reverse_loss_landscape)
                        classifier_loss = classifier_loss_dict['loss'].div(
                            denom)  # a little trick to avoid gradient overflow with fp16  # should be -
                        classifier_loss_data = classifier_loss_dict['data'] if classifier_loss_dict['data'] is not None else 0
                        # loss_data += classifier_loss_data

                        if not self.opt.freeze_language_classifier:
                            if self.cuda:
                                with amp.scale_loss(classifier_loss, optimizer) as scaled_loss:
                                    scaled_loss.backward()
                            else:
                                classifier_loss.backward()

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory on GPU , skipping batch')
                        oom = True
                        torch.cuda.empty_cache()
                        loss = 0
                        if opt.streaming:  # reset stream in this case ...
                            streaming_state = self.model.init_stream()
                    else:
                        raise e

                if loss != loss or (has_classifier_loss and classifier_loss != classifier_loss):
                    # catching NAN problem
                    oom = True
                    self.model.zero_grad()
                    self.optim.zero_grad()
                    num_accumulated_words = 0
                    num_accumulated_sents = 0

                if not oom:
                    src_size = batch.src_size
                    tgt_size = batch.tgt_size

                    counter = counter + 1
                    num_accumulated_words += tgt_size
                    num_accumulated_sents += batch_size

                    #   We only update the parameters after getting gradients from n mini-batches
                    update_flag = False
                    if 0 < opt.batch_size_update <= num_accumulated_words:
                        update_flag = True
                    elif counter >= opt.update_frequency and 0 >= opt.batch_size_update:
                        update_flag = True
                    elif i == n_samples - 1:  # update for the last minibatch
                        update_flag = True

                    if update_flag:
                        grad_denom = 1 / denom
                        if self.opt.normalize_gradient:
                            grad_denom = num_accumulated_words / denom
                        normalize_gradients(amp.master_params(optimizer), grad_denom)
                        # Update the parameters.
                        if self.opt.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.opt.max_grad_norm)
                        self.optim.step(grad_denom=grad_denom)
                        self.optim.zero_grad()
                        self.model.zero_grad()
                        counter = 0
                        num_accumulated_words = 0
                        num_accumulated_sents = 0
                        num_updates = self.optim._step

                        if opt.save_every > 0 and num_updates % opt.save_every == -1 % opt.save_every:
                            valid_loss, _ = self.eval(self.valid_data)
                            valid_ppl = math.exp(min(valid_loss, 100))
                            print('Validation perplexity: %g' % valid_ppl)

                            ep = float(epoch) - 1. + ((float(i) + 1.) / n_samples)

                            self.save(ep, valid_ppl, batch_order=batch_order, iteration=i)

                    num_words = tgt_size
                    report_loss += loss_data
                    report_classifier_loss += classifier_loss_data if self.opt.language_classifier else 0
                    report_tgt_words += num_words
                    report_src_words += src_size
                    total_loss += loss_data
                    total_words += num_words
                    total_tokens += batch.get('target_output').nelement()
                    total_non_pads += batch.get('target_output').ne(onmt.constants.PAD).sum().item()
                    optim = self.optim
                    batch_efficiency = total_non_pads / total_tokens

                    if b == 0 and (i == 0 or (i % opt.log_interval == -1 % opt.log_interval)):
                        print(("Epoch %2d, %5d/%5d; ; ppl: %6.2f ; adv loss: %6.6f ; lr: %.7f ; num updates: %7d " +
                               "%5.0f src tok/s; %5.0f tgt tok/s; %s elapsed") %
                              (epoch, i + 1, len(train_data),
                               math.exp(report_loss / report_tgt_words),
                               math.exp(math.log(max(0.000001, report_classifier_loss)) - math.log(report_src_words)),
                               optim.getLearningRate(),
                               optim._step,
                               report_src_words / (time.time() - start),
                               report_tgt_words / (time.time() - start),
                               str(datetime.timedelta(seconds=int(time.time() - self.start_time)))))

                        report_loss, report_tgt_words = 0, 0
                        report_classifier_loss = 0.000001
                        report_src_words = 0
                        start = time.time()

        return total_loss / total_words

    # def run(self, save_file=None):
    def run(self, checkpoint=None):

        opt = self.opt
        model = self.model
        optim = self.optim

        # Try to load the save_file
        # checkpoint = None
        # if save_file:
        #     checkpoint = torch.load(save_file, map_location=lambda storage, loc: storage)

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model'])
            prev_opt = checkpoint['opt'] if 'opt' in checkpoint else None

            if not opt.reset_optim:
                self.optim.load_state_dict(checkpoint['optim'])
                if prev_opt is not None and hasattr(prev_opt, "fp16_mixed"):
                    # Only load amp information if the mode is the same
                    # Maybe its better to change between optimization mode?
                    if opt.fp16_mixed == prev_opt.fp16_mixed and opt.fp16 == prev_opt.fp16:
                        if 'amp' in checkpoint:
                            amp.load_state_dict(checkpoint['amp'])

                if 'batch_order' in checkpoint:
                    batch_order = checkpoint['batch_order']
                    iteration = checkpoint['iteration'] + 1
                else:
                    batch_order = None
                    iteration = 0
                opt.start_epoch = int(math.floor(float(checkpoint['epoch'] + 1)))

                resume = True
                if len(self.additional_data) > 0:
                    if 'additional_batch_order' in checkpoint:
                        self.additional_batch_order = checkpoint['additional_batch_order']
                        self.additional_data_iteration = checkpoint['additional_data_iteration']
                    else:
                        self.init_additional_data()
            else:
                batch_order = None
                iteration = 0
                resume = False
                self.init_additional_data()

            del checkpoint['model']
            del checkpoint['optim']
            del checkpoint
        else:
            batch_order = None
            iteration = 0
            print('Initializing model parameters')
            init_model_parameters(model, opt)
            resume = False
            self.init_additional_data()

        if opt.load_encoder_from:
            self.load_encoder_weight(opt.load_encoder_from)

        valid_loss, valid_adv_loss = self.eval(self.valid_data)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g, adv loss: %6.6f' % (valid_ppl, valid_adv_loss))

        self.start_time = time.time()

        for epoch in range(opt.start_epoch, opt.start_epoch + opt.epochs):
            print('')

            #  (1) train for one epoch on the training set
            train_loss = self.train_epoch(epoch, resume=resume,
                                          batch_order=batch_order,
                                          iteration=iteration)
            train_ppl = math.exp(min(train_loss, 100))
            print('Train perplexity: %g' % train_ppl)

            #  (2) evaluate on the validation set
            valid_loss, valid_adv_loss = self.eval(self.valid_data)
            valid_ppl = math.exp(min(valid_loss, 100))
            print('Validation perplexity: %g, adv loss: %6.6f' % (valid_ppl, valid_adv_loss))

            self.save(epoch, valid_ppl)
            batch_order = None
            iteration = None
            resume = False

    def init_additional_data(self):
        self.additional_batch_order = []
        self.additional_data_iteration = []
        for i in range(len(self.additional_data)):
            self.additional_data_iteration.append(0)
            self.additional_data[i].shuffle()
            self.additional_batch_order.append(self.additional_data[i].create_order())


