from ctypes import alignment
from fileinput import close
import os
import csv
import sys
import argparse
import tqdm
import random
from mosestokenizer import MosesTokenizer

parser = argparse.ArgumentParser(description='extract_train_from_mustshe.py')
parser.add_argument('-data_dir_path', required=True, default=None)

def extract_train_set_from_tsv():

    opt = parser.parse_args()
    file_path = opt.data_dir_path

    # out_path = file_path + "train/"
    out_path = file_path

    tsv_file_es = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.es_v1.2.tsv"), encoding='utf-8')
    tsv_file_fr = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.fr_v1.2.tsv"), encoding='utf-8')
    tsv_file_it = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.it_v1.2.tsv"), encoding='utf-8')

    map_es = create_language_pair_dict_with_add_info(tsv_file_es)
    map_fr = create_language_pair_dict_with_add_info(tsv_file_fr)
    map_it = create_language_pair_dict_with_add_info(tsv_file_it)

    mustshe_exclude_path = file_path + "/correct_ref/"

    for gender_set in ["all", "feminine", "masculine"]:
        es_ex_file = open(os.path.join(mustshe_exclude_path, gender_set, "es-en.s"), encoding='utf-8')
        en_es_ex_file = open(os.path.join(mustshe_exclude_path, gender_set, "es-en.t"), encoding='utf-8')
        fr_ex_file = open(os.path.join(mustshe_exclude_path, gender_set, "fr-en.s"), encoding='utf-8')
        en_fr_ex_file = open(os.path.join(mustshe_exclude_path, gender_set, "fr-en.t"), encoding='utf-8')
        it_ex_file = open(os.path.join(mustshe_exclude_path, gender_set, "it-en.s"), encoding='utf-8')
        en_it_ex_file = open(os.path.join(mustshe_exclude_path, gender_set, "it-en.t"), encoding='utf-8')

        map_es_ex = create_language_pair_dict(en_es_ex_file, es_ex_file)
        map_fr_ex = create_language_pair_dict(en_fr_ex_file, fr_ex_file)
        map_it_ex = create_language_pair_dict(en_it_ex_file, it_ex_file)
    
        map_es_cleaned = remove_test_instances(map_es_ex, map_es[gender_set])
        map_fr_cleaned = remove_test_instances(map_fr_ex, map_fr[gender_set])
        map_it_cleaned = remove_test_instances(map_it_ex, map_it[gender_set])

        export_train_instances(map_es_cleaned, sl="en", tl="es", out_path=out_path, gender_set=gender_set)
        export_train_instances(map_fr_cleaned, sl="en", tl="fr", out_path=out_path, gender_set=gender_set)
        export_train_instances(map_it_cleaned, sl="en", tl="it", out_path=out_path, gender_set=gender_set)


def create_language_pair_dict_with_add_info(in_file):
    read_tsv_tl = csv.reader(in_file, delimiter="\t")
    next(read_tsv_tl)
    map_tl = {
        "all": {
            "src": [],
            "ref": [],
            "speaker": [],
            "category": [],
            "gterms": [],
            "gender_sentence_label": [],
            "gender_word_labels": [],
            "gender_tok_labels": []
        },
        "feminine": {
            "src": [],
            "ref": [],
            "speaker": [],
            "category": [],
            "gterms": [],
            "gender_sentence_label": [],
            "gender_word_labels": [],
            "gender_tok_labels": []
        },
        "masculine": {
            "src": [],
            "ref": [],
            "speaker": [],
            "category": [],
            "gterms": [],
            "gender_sentence_label": [],
            "gender_word_labels": [],
            "gender_tok_labels": []
        }
    }

    for row in read_tsv_tl:
        src = row[4]
        ref = row[5]
        speaker_gender = row[8]
        category = row[9]
        gender_terms = row[12].split(';')

        gender_terms_cr = []
        for gt in gender_terms:
            cr_wr = gt.split(' ')
            gender_terms_cr.append(cr_wr[0])

        # note, all below are 'correct_ref'
        map_tl["all"]["src"].append(src)
        map_tl["all"]["ref"].append(ref)
        map_tl["all"]["speaker"].append(speaker_gender)
        map_tl["all"]["category"].append(category)
        map_tl["all"]["gterms"].append(gender_terms_cr)

        if "F" in category:
            map_tl["feminine"]["src"].append(src)
            map_tl["feminine"]["ref"].append(ref)
            map_tl["feminine"]["speaker"].append(speaker_gender)
            map_tl["feminine"]["category"].append(category)
            map_tl["feminine"]["gterms"].append(gender_terms_cr)
            map_tl["feminine"]["gender_sentence_label"].append('2') # 'f'
            map_tl["all"]["gender_sentence_label"].append('2')  # 'f'
        
        if "M" in category:
            map_tl["masculine"]["src"].append(src)
            map_tl["masculine"]["ref"].append(ref)
            map_tl["masculine"]["speaker"].append(speaker_gender)
            map_tl["masculine"]["category"].append(category)
            map_tl["masculine"]["gterms"].append(gender_terms_cr)
            map_tl["masculine"]["gender_sentence_label"].append('1') # 'm'
            map_tl["all"]["gender_sentence_label"].append('1') # 'm'

    return map_tl


def create_language_pair_dict(sl_file, tl_file):
    map_sl_tl = dict(zip(sl_file, tl_file))
    return map_sl_tl


def remove_test_instances(en_tl_ex, en_tl):
    for en in en_tl_ex.keys():
        if en in en_tl["src"]:
            idx = en_tl["src"].index(en)
            del en_tl["src"][idx]
            del en_tl["ref"][idx]
            del en_tl["speaker"][idx]
            del en_tl["category"][idx]
            del en_tl["gterms"][idx]
            del en_tl["gender_sentence_label"][idx]
    return en_tl
    

def export_train_instances(map, sl, tl, out_path, gender_set):

    map = word_level_gender_labels(map)

    # train/valid split
    num_valid = round(len(map["src"]) * 0.075)

    valid_indices = list(range(0, len(map["src"])-1))
    random.shuffle(valid_indices)
    valid_indices = valid_indices[:num_valid]

    sl_out_file = open(os.path.join(out_path, f"train/{gender_set}/{sl}-{tl}.s"), "w", encoding='utf-8')
    tl_out_file = open(os.path.join(out_path, f"train/{gender_set}/{tl}-{sl}.s"), "w", encoding='utf-8')
    # ctg_out_file = open(os.path.join(out_path, f"train/{gender_set}/annotation/{tl}_category.s"), "w", encoding='utf-8')
    # spk_out_file = open(os.path.join(out_path, f"train/{gender_set}/annotation/{tl}_speaker.s"), "w", encoding='utf-8')
    # gtrms_out_file = open(os.path.join(out_path, f"train/{gender_set}/annotation/{tl}_gterms.s"), "w", encoding='utf-8')
    gndr_out_file = open(os.path.join(out_path, f"train/{gender_set}/gen_label/sent/{tl}.s"), "w", encoding='utf-8') # sentence gender label
    gndr_labels_out_file = open(os.path.join(out_path, f"train/{gender_set}/gen_label/word/{tl}.s"), "w", encoding='utf-8') # word gender labels
    gndr_tok_labels_out_file = open(os.path.join(out_path, f"train/{gender_set}/gen_label/tok/{tl}.s"), "w", encoding='utf-8') # word gender labels

    # alignment_out_file = open(os.path.join(out_path, f"valid/{gender_set}/annotation/{tl}_align.s"), "w", encoding='utf-8')

    sl_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/{sl}-{tl}.s"), "w", encoding='utf-8')
    tl_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/{tl}-{sl}.s"), "w", encoding='utf-8')
    # ctg_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/annotation/{tl}_category.s"), "w", encoding='utf-8')
    # spk_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/annotation/{tl}_speaker.s"), "w", encoding='utf-8')
    # gtrms_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/annotation/{tl}_gterms.s"), "w", encoding='utf-8')
    gndr_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/gen_label/sent/{tl}.s"), "w", encoding='utf-8') # sentence gender label
    gndr_labels_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/gen_label/word/{tl}.s"), "w", encoding='utf-8')  # word gender labels
    gndr_tok_labels_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/gen_label/tok/{tl}.s"), "w", encoding='utf-8')  # word gender labels
    
    # alignment_out_file_v = open(os.path.join(out_path, f"valid/{gender_set}/annotation/{tl}_align.s"), "w", encoding='utf-8')

    for i, sl in enumerate(map["src"]):
        if i in valid_indices:
            sl_out_file_v.write(sl + '\n')
        else:
            sl_out_file.write(sl + '\n')
    for i, tl in enumerate(map["ref"]):
        if i in valid_indices:
            tl_out_file_v.write(tl + '\n')
        else:
            tl_out_file.write(tl + '\n')
    # for i, ctg in enumerate(map["category"]):
    #     if i in valid_indices:
    #         ctg_out_file_v.write(ctg + '\n')
    #     else:
    #         ctg_out_file.write(ctg + '\n')
    # for i, spk in enumerate(map["speaker"]):
    #     if i in valid_indices:
    #         spk_out_file_v.write(spk + '\n')
    #     else:
    #         spk_out_file.write(spk + '\n')
    # for i, gterms in enumerate(map["gterms"]):
    #     if i in valid_indices:
    #         for term in gterms:
    #             gtrms_out_file_v.write(term + ' ')
    #         gtrms_out_file_v.write('\n')    
    #     else:
    #         for term in gterms:
    #             gtrms_out_file.write(term + ' ')
    #         gtrms_out_file.write('\n')
    for i, gndr in enumerate(map["gender_sentence_label"]):
        if i in valid_indices:
            gndr_out_file_v.write(gndr + '\n')
        else:
            gndr_out_file.write(gndr + '\n')
    for i, gndr in enumerate(map["gender_word_labels"]):
        if i in valid_indices:
            gndr_labels_out_file_v.write(gndr + '\n')
        else:
            gndr_labels_out_file.write(gndr + '\n')

    # tokenizer_sl = MosesTokenizer(f"{sl}")
    # tokenizer_tl = MosesTokenizer(f"{tl}")
    # i = 0
    # for src, tgt in tqdm.tqdm(zip(map["src"], map["ref"])):
    #     tokenized_src = tokenizer_sl(src)
    #     tokenized_tgt = tokenizer_tl(tgt)
    #     for tok in tokenized_src:
    #         if i in valid_indices:
    #             alignment_out_file_v.write(tok + " ")
    #         else:
    #             alignment_out_file.write(tok + " ")
    #     if i in valid_indices:
    #         alignment_out_file_v.write("||| ")
    #     else:
    #         alignment_out_file.write("||| ")
    #     for tok in tokenized_tgt:
    #         if i in valid_indices:
    #             alignment_out_file_v.write(tok + " ")
    #         else:
    #             alignment_out_file.write(tok + " ")
    #     if i in valid_indices:
    #         alignment_out_file_v.write("\n")
    #     else:
    #         alignment_out_file.write("\n")
        
    #     i += 1


    # tokenizer_sl = MosesTokenizer(f"{sl}")
    # for i, sent in enumerate(map["src"]):
    #     s_tok = tokenizer_sl(sent)
    #     for tok in 



    # TODO
    # tokenizer_sl = MosesTokenizer(f"{sl}")
    # for i, sent in enumerate(map["src"]):
    #     # for each sentence...
    #     s_tok_labels = ""
    #     words = sent.split()
    #     for j, w in enumerate(words):
    #         # for each word...
    #         w_gender = "0" # neuter
    #         print(w)
    #         if w in map["gterms"][i]:
    #             if map["category"][i][1] == "M":
    #                 # masculine
    #                 w_gender = "1"
    #             else:
    #                 # feminine
    #                 w_gender = "2"

    #         w_tok = tokenizer_sl(w)
    #         w_tok_labels = len(w_tok) * [w_gender]
    #         for k, l in enumerate(w_tok_labels):
    #             # for each token...
    #             s_tok_labels += f"{l}"
    #             if i < len(words) - 1:
    #                 # not last word in sentence
    #                 s_tok_labels += " "
    #             else:
    #                 if k < len(w_tok_labels) - 1:
    #                     # not the last tok in last word
    #                     s_tok_labels += " "
    #                 else:
    #                     continue
                    
    #         # print(s_tok_labels)
            
    #     map["gender_tok_labels"].append(s_tok_labels)

    # for i, gndr in enumerate(map["gender_tok_labels"]):
    #     if i in valid_indices:
    #         gndr_tok_labels_out_file_v.write(gndr + '\n')
    #     else:
    #         gndr_tok_labels_out_file.write(gndr + '\n')

    # sl_out_file.close()
    # tl_out_file.close()
    # ctg_out_file.close()
    # spk_out_file.close()
    # gtrms_out_file.close()
    # alignment_out_file.close()


def word_level_gender_labels(map):

    n_macsuline = 0
    n_feminine = 0
    for i, sentence in enumerate(map["ref"]):
        word_labels = ""
        words = sentence.split()
        for j, word in enumerate(words):
            punctuation_marks = [".", ",", "!", "?", ":", ";", "¿", "¡", "\"", "\n"]
            for mark in punctuation_marks:
                word = word.replace(mark, "")
            word_label = "0" # neuter
            if word in map["gterms"][i]:
                gender = map["category"][i][1]
                if gender == "M":
                    # masculine
                    word_label = "1"
                    n_macsuline += 1
                else:
                    # feminine
                    word_label = "2"
                    n_feminine += 1
            
            word_labels += word_label
            if j < len(words) - 1:
                word_labels += " "
        map["gender_word_labels"].append(word_labels)

    print(f"In this gender set: {n_macsuline} masculine and {n_feminine} feminine words.")

    return map


if __name__ == '__main__':
    extract_train_set_from_tsv()