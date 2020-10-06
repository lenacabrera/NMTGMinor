from __future__ import division

import math
import torch
import torch.utils.data
from collections import defaultdict
import onmt
from onmt.speech.Augmenter import Augmenter
from onmt.modules.dropout import switchout

"""
Data management for sequence-to-sequence models
Two basic classes: 
- Batch stores the input / output sequences, grouped into tensors with the same length (by padding)
- Dataset stores all of the data and 
"""


class Batch(object):
    # An object to manage the data within a minibatch
    def __init__(self, src_data, tgt_data=None,
                 src_lang_data=None, tgt_lang_data=None,
                 src_type='text',
                 src_align_right=False, tgt_align_right=False,
                 augmenter=None, upsampling=False,
                 merge=False, token_level_lang=False, **kwargs):
        """
        :param src_data: list of source tensors
        :param tgt_data: list of target tensors
        :param src_atb_data: list of attributes/features for the source (TB finished)
        :param tgt_atb_data: list of attributes/features for the target (TB finished)
        :param src_type: text or audio
        :param src_align_right: if the source sequences are aligned to the right
        :param tgt_align_right: if the target sequences are aligned to the right
        (default False and maybe never changed unless new models need)
        :param reshape_speech: the number of frames to be reshaped
        :param augmenter: using augmentation for speech
        :param merge: if the two sequences are going to be merged for Relative Transformer
        """

        self.tensors = defaultdict(lambda: None)
        self.has_target = False
        self.src_type = src_type
        # self.upsampling = kwargs.get('upsampling', False)
        self.upsampling = upsampling
        self.feature_size = kwargs.get('feature_size', 40)
        # self.reshape_speech = reshape_speech
        self.src_align_right = src_align_right
        if merge:
            self.src_align_right = True

        self.tgt_align_right = tgt_align_right

        if src_data is not None:
            self.tensors['source'], self.tensors['source_pos'], self.src_lengths = \
                                                                    self.collate(src_data,
                                                                                 align_right=self.src_align_right,
                                                                                 type=self.src_type,
                                                                                 augmenter=augmenter)
            self.tensors['source'] = self.tensors['source'].transpose(0, 1).contiguous()
            if self.tensors['source_pos'] is not None:
                self.tensors['source_pos'] = self.tensors['source_pos'].transpose(0, 1)
            self.tensors['src_length'] = torch.LongTensor(self.src_lengths)
            self.src_lengths = self.tensors['src_length']
            self.src_size = sum(self.src_lengths)
        else:
            self.src_size = 0

        if tgt_data is not None:
            target_full, target_pos, self.tgt_lengths = self.collate(tgt_data, align_right=self.tgt_align_right)
            target_full = target_full.t().contiguous()  # transpose BxT to TxB
            self.tensors['target'] = target_full
            self.tensors['target_input'] = target_full[:-1]
            self.tensors['target_output'] = target_full[1:]
            self.tensors['target_pos'] = target_pos.t().contiguous()[:-1]
            self.tensors['tgt_mask'] = self.tensors['target_output'].ne(onmt.constants.PAD)
            self.has_target = True
            self.tgt_size = sum([len(x) - 1 for x in tgt_data])

        else:
            self.tgt_size = 0

        self.size = len(src_data) if src_data is not None else len(tgt_data)

        if src_lang_data is not None:

            self.tensors['source_lang'] = torch.cat(src_lang_data).long()  # per sentence

            if token_level_lang:  # prepare token-level prediction targets
                sl_ = self.tensors['source_lang']  # sl_.shape[0]
                out_dims = (max(self.src_lengths).item(), sl_.shape[0])
                out_tensor = sl_.data.new(*out_dims).fill_(onmt.constants.PAD)
                for i, v in enumerate(sl_):
                    # lan ID starts with 0 but pred label values should start with 1 (due to padding)
                    out_tensor[:(self.src_lengths[i]), i] = v+1

                # uni, uni_cnt = torch.unique(out_tensor, return_counts=True)
                # print('=========================', uni, uni_cnt)
                self.tensors['targets_source_lang'] = out_tensor

        if tgt_lang_data is not None:
            self.tensors['target_lang'] = torch.cat(tgt_lang_data).long()

    def switchout(self, swrate, src_vocab_size, tgt_vocab_size):
        # Switch out function ... currently works with only source text data
        if self.src_type == 'text':
            self.tensors['source'] = switchout(self.tensors['source'], src_vocab_size, swrate, transpose=True)

        if self.has_target:
            self.tensors['target'] = switchout(self.tensors['target'], tgt_vocab_size, swrate, transpose=True, offset=1)
            target_full = self.tensors['target']
            self.tensors['target_input'] = target_full[:-1]
            self.tensors['target_output'] = target_full[1:]
            self.tensors['tgt_mask'] = self.tensors['target_output'].ne(onmt.constants.PAD)

    # down sampling the speech signal by simply concatenating n features (reshaping)
    def downsample(self, data):

        if self.reshape_speech == 0:
            return data

        else:
            concat = self.reshape_speech
            tensor_ = data.float()  # adding float because of fp16 data storage
            add = (concat - tensor_.size()[0] % concat) % concat
            z = torch.FloatTensor(add, tensor_.size()[1]).zero_()

            # adding an additional dimension as padding
            tensor_ = torch.cat((tensor_, z), 0)
            tensor_ = tensor_.reshape((int(tensor_.size()[0] / concat), tensor_.size()[1] * concat))

            return tensor_

    def augment_speech(self):

        return

    def collate(self, data, align_right=False, type="text", augmenter=None):
        """
        Assembling the individual sequences into one single tensor, included padding
        :param data: the list of sequences
        :param align_right: aligning the sequences w.r.t padding
        :param type: text or audio
        :param augmenter: for augmentation in audio models
        :return:
        """
        # initialize with batch_size * length
        if type == "text":
            lengths = [x.size(0) for x in data]
            positions = [torch.arange(length_) for length_ in lengths]
            max_length = max(lengths)
            tensor = data[0].new(len(data), max_length).fill_(onmt.constants.PAD)
            pos = tensor.new(*tensor.size()).fill_(0)

            for i in range(len(data)):
                data_length = data[i].size(0)
                offset = max_length - data_length if align_right else 0
                tensor[i].narrow(0, offset, data_length).copy_(data[i])
                pos[i].narrow(0, offset, data_length).copy_(positions[i])

            return tensor, pos, lengths

        elif type == "audio":

            # First step: on-the-fly processing for the samples
            # Reshaping: either downsampling or upsampling
            # On the fly augmentation
            samples = []

            for i in range(len(data)):
                sample = data[i]

                if augmenter is not None:
                    sample = augmenter.augment(sample)

                if self.upsampling:
                    sample = sample.view(-1, self.feature_size)

                samples.append(sample)

            # compute the lengths afte on-the-fly processing
            lengths = [x.size(0) for x in samples]

            max_length = max(lengths)

            # allocate data for the batch speech
            feature_size = samples[0].size(1)
            batch_size = len(data)

            # feature size + 1 because the last dimension is created for padding
            tensor = data[0].float().new(batch_size, max_length, feature_size + 1).fill_(onmt.constants.PAD)

            for i in range(len(samples)):
                sample = samples[i]

                data_length = sample.size(0)
                offset = max_length - data_length if align_right else 0

                tensor[i].narrow(0, offset, data_length).narrow(1, 1, sample.size(1)).copy_(sample)
                # in padding dimension: 0 is not padded, 1 is padded
                tensor[i].narrow(0, offset, data_length).narrow(1, 0, 1).fill_(1)

            return tensor, None, lengths
        else:
            raise NotImplementedError

    def get(self, name):
        if name in self.tensors:
            return self.tensors[name]
        else:
            return None

    def cuda(self, fp16=False):
        """
        Send the minibatch data into GPU. Old-fashioned without the 'device' control
        :param fp16:
        :return: None
        """
        for key, tensor in self.tensors.items():
            if isinstance(tensor, dict):
                for k in tensor:
                    v = tensor[k]
                    tensor[k] = v.cuda()
            elif tensor is not None:
                if tensor.type() == "torch.FloatTensor" and fp16:
                    self.tensors[key] = tensor.half()
                self.tensors[key] = self.tensors[key].cuda()
            else:
                continue


class Dataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data,
                 src_langs=None, tgt_langs=None,
                 batch_size_words=2048,
                 data_type="text", batch_size_sents=128,
                 multiplier=1, sorting=False,
                 augment=False,
                 src_align_right=False, tgt_align_right=False,
                 verbose=False, cleaning=False,
                 token_level_lang=False,
                 **kwargs):
        """
        :param src_data: List of tensors for the source side (1D for text, 2 or 3Ds for other modalities)
        :param tgt_data: List of tensors (1D text) for the target side (already padded with <s> and </s>
        :param src_langs: Source languages (list of one-tensors)
        :param tgt_langs: Target Languages (list of one-tensors)
        :param batch_size_words: Maximum number of words in the minibatch (MB can't have more than this)
        :param data_type: Text or Audio
        :param batch_size_sents: Maximum number of sequences in the minibatch (MB can't have more than this)
        :param multiplier: The number of sequences must divide by this number (for fp16 when multiplier=8)
        :param reshape_speech: Put N frames together to reduce the length (this might be done already in preprocessing)
        :param augment: Speech Augmentation (currently only spec augmentation is implemented)
        """

        """
        For alignment, the right-aligned data looks like:
        P P P P D D D D
        P P D D D D D D
        P P P P P D D D
        P P P D D D D D
        This can affect positional encoding (whose implementation is not consistent w.r.t padding)
        For models with absolute positional encoding, src and tgt should be aligned left (This is default)
        For models with relative positional encoding, src should be right and tgt should be left
        """

        self.src = src_data
        self._type = data_type
        self.src_align_right = src_align_right
        if self.src_align_right and verbose:
            print("* Source sentences aligned to the right side.")
        self.tgt_align_right = tgt_align_right
        self.upsampling = kwargs.get('upsampling', False)

        self.max_src_len = kwargs.get('max_src_len', None)
        self.max_tgt_len = kwargs.get('max_tgt_len', 256)

        if self.max_src_len is None:
            if self._type == 'text':
                self.max_src_len = 256
            else:
                self.max_src_len = 1024

        # self.reshape_speech = reshape_speech
        if tgt_data:
            self.tgt = tgt_data

            if src_data:
                assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None

        # Remove the sentences that are empty
        if cleaning:
            cleaned_src = []
            cleaned_tgt = []
            n_remove = 0

            for (src_tensor, tgt_tensor) in zip(self.src, self.tgt):

                src_size = src_tensor.size(0)
                tgt_size = tgt_tensor.size(0)

                if 0 < src_size < self.max_src_len and 2 < tgt_size < self.max_tgt_len:
                    cleaned_src.append(src_tensor)
                    cleaned_tgt.append(tgt_tensor)
                else:
                    n_remove += 1

            self.src = cleaned_src
            self.tgt = cleaned_tgt
            if verbose:
                print("* Removed %d empty sentences" % n_remove)

        if verbose:
            print("* Loaded data with %d sentences." % len(self.src))

        # sort data to have efficient mini-batching during training
        if sorting:
            assert self.src is not None
            assert self.tgt is not None

            src_sizes = [data.size(0) for data in self.src]
            tgt_sizes = [data.size(0) for data in self.tgt]
            orders = range(len(self.src))

            z = zip(src_sizes, tgt_sizes, orders)

            if self._type == 'text':
                # For machine translation, sort by source first, and then target
                sorted_z = sorted(sorted(z, key=lambda x: x[0]), key=lambda x: x[1])

            elif self._type == 'audio':
                sorted_z = sorted(sorted(z, key=lambda x: x[1]), key=lambda x: x[0])

            sorted_order = [z_[2] for z_ in sorted_z]

            self.src = [self.src[i] for i in sorted_order]
            self.tgt = [self.tgt[i] for i in sorted_order]

        self.src_langs = src_langs
        self.tgt_langs = tgt_langs

        if self.src_langs is not None and self.tgt_langs is not None:
            assert(len(src_langs) == len(tgt_langs))

        # In "bilingual" case, the src_langs only contains one single vector
        # Which is broadcasted to batch_size
        if len(src_langs) <= 1:
            self.bilingual = True
        else:
            self.bilingual = False
            if sorting:
                self.src_langs = [self.src_langs[i] for i in sorted_order]
                self.tgt_langs = [self.tgt_langs[i] for i in sorted_order]
        print('bilingual', self.bilingual)  # TODO: true?

        self.fullSize = len(self.src) if self.src is not None else len(self.tgt)

        # maximum number of tokens in a mb
        self.batch_size_words = batch_size_words

        # maximum sequences in a mb
        self.batch_size_sents = batch_size_sents

        # the actual batch size must divide by this multiplier (for fp16 it has to be 4 or 8)
        self.multiplier = multiplier

        # by default: count the amount of padding when we group mini-batches
        self.pad_count = True

        # group samples into mini-batches
        self.batches = []
        self.num_batches = 0
        self.allocate_batch()

        self.cur_index = 0
        self.batchOrder = None

        if augment:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None

        self.token_level_lang = token_level_lang

    def size(self):

        return self.fullSize

    def switchout(self, batch):

        pass

    # This function allocates the mini-batches (grouping sentences with the same size)
    def allocate_batch(self):

        cur_batch = []
        cur_batch_size = 0
        cur_batch_sizes = []

        def oversize_(cur_batch, sent_size):

            if len(cur_batch) >= self.batch_size_sents:
                return True

            if not self.pad_count:
                if cur_batch_size + sent_size > self.batch_size_words:
                    return True
            else:
                if len(cur_batch_sizes) == 0:
                    return False

                if (max(max(cur_batch_sizes), sent_size)) * (len(cur_batch) + 1) > self.batch_size_words:
                    return True
            return False

        i = 0
        while i < self.fullSize:

            if self.tgt is not None and self.src is not None:
                sentence_length = max(self.tgt[i].size(0) - 1, self.src[i].size(0))
                # print(sentence_length)
            elif self.tgt is not None:
                sentence_length = self.tgt[i].size(0) - 1
            else:
                sentence_length = self.src[i].size(0)

            oversized = oversize_(cur_batch, sentence_length)
            # if the current item makes the batch exceed max size
            # then we create a new batch
            if oversized:
                # cut-off the current list to fit the multiplier
                current_size = len(cur_batch)
                scaled_size = max(
                    self.multiplier * (current_size // self.multiplier),
                    current_size % self.multiplier)

                batch_ = cur_batch[:scaled_size]
                self.batches.append(batch_)  # add this batch into the batch list

                cur_batch = cur_batch[scaled_size:]  # reset the current batch
                cur_batch_sizes = cur_batch_sizes[scaled_size:]
                cur_batch_size = sum(cur_batch_sizes)

            cur_batch.append(i)
            cur_batch_size += sentence_length
            cur_batch_sizes.append(sentence_length)

            i = i + 1

        # catch the last batch
        if len(cur_batch) > 0:
            self.batches.append(cur_batch)

        self.num_batches = len(self.batches)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        """
        :param index: the index of the mini-batch in the list
        :return: Batch
        """
        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)

        batch_ids = self.batches[index]
        if self.src:
            src_data = [self.src[i] for i in batch_ids]
        else:
            src_data = None

        if self.tgt:
            tgt_data = [self.tgt[i] for i in batch_ids]
        else:
            tgt_data = None

        src_lang_data = None
        tgt_lang_data = None

        if self.bilingual:
            if self.src_langs is not None:
                src_lang_data = [self.src_langs[0]]  # should be a tensor [0]
            if self.tgt_langs is not None:
                tgt_lang_data = [self.tgt_langs[0]]  # should be a tensor [1]
        else:
            if self.src_langs is not None:
                src_lang_data = [self.src_langs[i] for i in batch_ids]
            if self.tgt_langs is not None:
                tgt_lang_data = [self.tgt_langs[i] for i in batch_ids]

        batch = Batch(src_data, tgt_data=tgt_data,
                      src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
                      src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
                      src_type=self._type,
                      augmenter=self.augmenter, upsampling=self.upsampling,
                      token_level_lang=self.token_level_lang)

        return batch

    def __len__(self):
        return self.num_batches

    # genereate a new batch - order (static)
    def create_order(self, random=True):

        if random:
            self.batchOrder = torch.randperm(self.num_batches)
        else:
            self.batchOrder = torch.arange(self.num_batches).long()

        self.cur_index = 0

        return self.batchOrder

    # return the next batch according to the iterator
    def next(self, curriculum=False, reset=True, split_sizes=1):

        # reset iterator if reach data size limit
        if self.cur_index >= self.num_batches:
            if reset:
                self.cur_index = 0
            else:
                return None

        if curriculum or self.batchOrder is None:
            batch_index = self.cur_index
        else:
            batch_index = self.batchOrder[self.cur_index]

        batch = self[batch_index]

        # move the iterator one step
        self.cur_index += 1

        return [batch]

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])

    def set_index(self, iteration):

        assert (0 <= iteration < self.num_batches)
        self.cur_index = iteration

#
# # LANGUAGE MODEL DATASET AND DATAHOLDER
# class LMBatch(Batch):
#
#     def __init__(self, input, target=None):
#         self.tensors = defaultdict(lambda: None)
#
#         self.tensors['target_input'] = input  # T x B
#         self.tensors['target_output'] = target  # T x B or None
#
#         # batch size
#         self.size = input.size(1)
#         self.length = input.size(0)
#
#         self.tgt_size = self.size * self.length
#         self.src_size = 0
#
#     def collate(self, **kwargs):
#         raise NotImplementedError
#
#
# class LanguageModelDataset(Dataset):
#
#     def __init__(self, data, batch_size_sents=128, seq_length=128):
#
#         self.data = data
#
#         self.batch_size_sents = batch_size_sents
#
#         self.seq_length = seq_length
#
#         # group samples into mini batches
#         self.num_batches = 0
#         self.allocate_batch()
#
#         self.fullSize = self.num_batches
#         # self.cur_index = 0
#         # self.batchOrder = None
#
#     def allocate_batch(self):
#
#         nsequence = self.data.size(0) // self.batch_size_sents
#
#         self.data = self.data.narrow(0, 0, nsequence * self.batch_size_sents)
#
#         # Evenly divide the data across the bsz batches.
#         self.data = self.data.view(self.batch_size_sents, -1).t().contiguous()
#
#         # self.num_steps = nbatch - 1
#
#         self.num_batches = math.ceil((self.data.size(0) - 1) / self.seq_length)
#
#     # genereate a new batch - order (static)
#     def create_order(self, random=False):
#
#         # For language model order shouldn't be random
#         if random:
#             self.batchOrder = torch.randperm(self.num_batches)
#         else:
#             self.batchOrder = torch.arange(self.num_batches).long()
#
#         self.cur_index = 0
#
#         return self.batchOrder
#
#     # return the next batch according to the iterator
#     # for language model
#     def next(self, curriculum=True, reset=True, split_sizes=1):
#
#         # reset iterator if reach data size limit
#         if self.cur_index >= self.num_batches:
#             if reset:
#                 self.cur_index = 0
#             else:
#                 return None
#
#         batch_index = self.cur_index
#
#         seq_len = self.seq_length
#
#         top_index = min(batch_index + seq_len, self.data.size(0) - 1)
#
#         batch = LMBatch(self.data[batch_index:top_index], target=self.data[batch_index + 1:top_index + 1])
#
#         # move the iterator one step
#         self.cur_index += seq_len
#
#         return [batch]