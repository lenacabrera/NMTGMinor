import os
import csv
import sys
from enum import Enum

from regex import P

def extract_from_tsv(data_dir_path,correct_ref_dir_name, wrong_ref_dir_name):

    class MODE(Enum):
        PARA_ID = 0          # parallel data, keep only identical sentences from orig data files
        PARA_POSTPROC = 1    # parallel data, non-identical sentences with same content are made identical (post-processed)
        NONPARA = 2          # non-parallel data, preprocess for each monolingual file separately

    mode = MODE.PARA_ID
    print("mode: ", mode.name)

    file_path = data_dir_path
    c_ref_dir = file_path + correct_ref_dir_name + "/"
    w_ref_dir = file_path + wrong_ref_dir_name + "/"

    # es
    tsv_file_es = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.es_v1.2.tsv"), encoding='utf-8')
    read_tsv_es = csv.reader(tsv_file_es, delimiter="\t")
    next(read_tsv_es)

    map_es = {}

    if mode.name == MODE.NONPARA.name:
        en_es_c = open(c_ref_dir + "en-es.en", "w", encoding='utf-8')
        es_en_c = open(c_ref_dir + "es-en.es", "w", encoding='utf-8')
        en_es_w = open(w_ref_dir + "en-es.en", "w", encoding='utf-8')
        es_en_w = open(w_ref_dir + "es-en.es", "w", encoding='utf-8')

    for row in read_tsv_es:
        src = row[4]
        ref = row[5]
        ref_wrong = row[6]
        speaker_gender = row[8]
        category = row[9]
        gender_terms = row[12]
        map_es[row[0]] = [src, ref, ref_wrong, speaker_gender, category, gender_terms]
        if mode.name == MODE.NONPARA.name:
            en_es_c.write(src + '\n')
            es_en_c.write(ref + '\n')
            en_es_w.write(src + '\n')
            es_en_w.write(ref_wrong + '\n')

    # it
    tsv_file_it = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.it_v1.2.tsv"), encoding='utf-8')
    read_tsv_it = csv.reader(tsv_file_it, delimiter="\t")
    next(read_tsv_it)

    map_it = {}

    if mode.name == MODE.NONPARA.name:
        en_it_c = open(c_ref_dir + "en-it.en", "w", encoding='utf-8')
        it_en_c = open(c_ref_dir + "it-en.it", "w", encoding='utf-8')
        en_it_w = open(w_ref_dir + "en-it.en", "w", encoding='utf-8')
        it_en_w = open(w_ref_dir + "it-en.it", "w", encoding='utf-8')

    for row in read_tsv_it:
        src = row[4]
        ref = row[5]
        ref_wrong = row[6]
        speaker_gender = row[8]
        category = row[9]
        gender_terms = row[12]
        map_it[row[0]] = [src, ref, ref_wrong, speaker_gender, category, gender_terms]
        if mode.name == MODE.NONPARA.name:
            en_it_c.write(src + '\n')
            it_en_c.write(ref + '\n')
            en_it_w.write(src + '\n')
            it_en_w.write(ref_wrong + '\n')

    # fr
    tsv_file_fr = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.fr_v1.2.tsv"), encoding='utf-8')
    read_tsv_fr = csv.reader(tsv_file_fr, delimiter="\t")
    next(read_tsv_fr)

    map_fr = {}

    if mode.name == MODE.NONPARA.name:
        en_fr_c = open(c_ref_dir + "en-fr.en", "w", encoding='utf-8')
        fr_en_c = open(c_ref_dir + "fr-en.fr", "w", encoding='utf-8')
        en_fr_w = open(w_ref_dir + "en-fr.en", "w", encoding='utf-8')
        fr_en_w = open(w_ref_dir + "fr-en.fr", "w", encoding='utf-8')

    for row in read_tsv_fr:
        src = row[4]
        ref = row[5]
        ref_wrong = row[6]
        ref = ref.replace(u'♪ ', '')  # replace special character
        speaker_gender = row[8]
        category = row[9]
        gender_terms = row[12]
        map_fr[row[0]] = [src, ref, ref_wrong, speaker_gender, category, gender_terms]
        if mode.name == MODE.NONPARA.name:
            en_fr_c.write(src + '\n')
            fr_en_c.write(ref + '\n')
            en_fr_w.write(src + '\n')
            fr_en_w.write(ref_wrong + '\n')

    # all
    if "PAR" in mode.name:
        with open(os.path.join(file_path, "tsv/", "MULTILINGUAL_v1.2.tsv"), encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            # skip header (IT;FR;ES;CATEGORY)
            csv_reader.__next__()

            par_es_c = open(os.path.join(c_ref_dir, "es_par.s"), "w", encoding='utf-8')
            par_it_c = open(os.path.join(c_ref_dir, "it_par.s"), "w", encoding='utf-8')
            par_fr_c = open(os.path.join(c_ref_dir, "fr_par.s"), "w", encoding='utf-8')
            par_en_c = open(os.path.join(c_ref_dir, "en_par.s"), "w", encoding='utf-8')

            par_es_w = open(os.path.join(w_ref_dir, "es_par.s"), "w", encoding='utf-8')
            par_it_w = open(os.path.join(w_ref_dir, "it_par.s"), "w", encoding='utf-8')
            par_fr_w = open(os.path.join(w_ref_dir, "fr_par.s"), "w", encoding='utf-8')
            par_en_w = open(os.path.join(w_ref_dir, "en_par.s"), "w", encoding='utf-8')

            es_add_info = open(os.path.join(file_path, "es_add.csv"), "w", encoding='utf-8')
            fr_add_info = open(os.path.join(file_path, "fr_add.csv"), "w", encoding='utf-8')
            it_add_info = open(os.path.join(file_path, "it_add.csv"), "w", encoding='utf-8')
            
            for row in csv_reader:
                it_id = row[0]
                fr_id = row[1]
                es_id = row[2]

                if it_id == "NULL" or fr_id == "NULL" or es_id == "NULL":
                    continue
                else:
                    es_src = map_es[es_id][0]
                    it_src = map_it[it_id][0]
                    fr_src = map_fr[fr_id][0]

                    if es_src == it_src == fr_src:
                        par_es_c.write(map_es[es_id][1] + "\n")
                        par_it_c.write(map_it[it_id][1] + "\n")
                        par_fr_c.write(map_fr[fr_id][1] + "\n")
                        par_en_c.write(es_src + "\n")

                        par_es_w.write(map_es[es_id][2] + "\n")
                        par_it_w.write(map_it[it_id][2] + "\n")
                        par_fr_w.write(map_fr[fr_id][2] + "\n")
                        par_en_w.write(es_src + "\n")

                        es_add_info.write(map_es[es_id][3] + ",")
                        es_add_info.write(map_es[es_id][4] + ",")
                        es_add_info.write(map_es[es_id][5] + "\n")

                        it_add_info.write(map_it[it_id][3] + ",")
                        it_add_info.write(map_it[it_id][4] + ",")
                        it_add_info.write(map_it[it_id][5] + "\n")

                        fr_add_info.write(map_fr[fr_id][3] + ",")
                        fr_add_info.write(map_fr[fr_id][4] + ",")
                        fr_add_info.write(map_fr[fr_id][5] + "\n")
                    else:
                        if mode.name == MODE.PARA_ID.name:
                            continue
                        elif mode.name == MODE.PARA_POSTPROC.name:
                            print("TODO: post-processing")
                        else:
                            pass


if __name__ == '__main__':
    args = sys.argv[1:]
    extract_from_tsv(args[0], args[1], args[2])
