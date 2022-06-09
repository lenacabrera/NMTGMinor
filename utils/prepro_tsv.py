import os
import csv
from enum import Enum

class MODE(Enum):
    PARA_ID = 0          # parallel data, keep only identical sentences from orig data files
    PARA_POSTPROC = 1    # parallel data, non-identical sentences with same content are made identical (post-processed)
    NONPARA = 2          # non-parallel data, preprocess for each monolingual file separately

mode = MODE.PARA_ID
print(mode.name)

file_path = os.path.dirname(os.path.abspath(__file__))

# es
tsv_file_es = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.es_v1.2.tsv"), encoding='utf-8')
read_tsv_es = csv.reader(tsv_file_es, delimiter="\t")
next(read_tsv_es)

map_es = {}

if mode.name == MODE.NONPARA.name:
    en_es = open("en-es.en", "w", encoding='utf-8')
    es_en = open("es-en.es", "w", encoding='utf-8')

for row in read_tsv_es:
    src = row[4]
    ref = row[5]
    map_es[row[0]] = [src, ref]
    if mode.name == MODE.NONPARA.name:
        en_es.write(src + '\n')
        es_en.write(ref + '\n')

# it
tsv_file_it = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.it_v1.2.tsv"), encoding='utf-8')
read_tsv_it = csv.reader(tsv_file_it, delimiter="\t")
next(read_tsv_it)

map_it = {}

if mode.name == MODE.NONPARA.name:
    en_it = open("en-it.en", "w", encoding='utf-8')
    it_en = open("it-en.it", "w", encoding='utf-8')

for row in read_tsv_it:
    src = row[4]
    ref = row[5]
    map_it[row[0]] = [src, ref]
    if mode.name == MODE.NONPARA.name:
        en_it.write(src + '\n')
        it_en.write(ref + '\n')

# fr
tsv_file_fr = open(os.path.join(file_path, "tsv/", "MONOLINGUAL.fr_v1.2.tsv"), encoding='utf-8')
read_tsv_fr = csv.reader(tsv_file_fr, delimiter="\t")
next(read_tsv_fr)

map_fr = {}

if mode.name == MODE.NONPARA.name:
    en_fr = open("en-fr.en", "w", encoding='utf-8')
    fr_en = open("fr-en.fr", "w", encoding='utf-8')

for row in read_tsv_fr:
    src = row[4]
    ref = row[5]
    ref = ref.replace(u'♪ ', '')  # replace special character
    map_fr[row[0]] = [src, ref]
    if mode.name == MODE.NONPARA.name:
        en_fr.write(src + '\n')
        fr_en.write(ref + '\n')

# all
if "PAR" in mode.name:
    with open(os.path.join(file_path, "tsv/", "MULTILINGUAL_v1.2.tsv"), encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        # skip header (IT;FR;ES;CATEGORY)
        csv_reader.__next__()

        par_es = open(os.path.join(file_path, "es_par.s"), "w", encoding='utf-8')
        par_it = open(os.path.join(file_path, "it_par.s"), "w", encoding='utf-8')
        par_fr = open(os.path.join(file_path, "fr_par.s"), "w", encoding='utf-8')
        par_en = open(os.path.join(file_path, "en_par.s"), "w", encoding='utf-8')
        
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
                    par_es.write(map_es[es_id][1] + "\n")
                    par_it.write(map_it[it_id][1] + "\n")
                    par_fr.write(map_fr[fr_id][1] + "\n")
                    par_en.write(es_src + "\n")
                else:
                    if mode.name == MODE.PARA_ID.name:
                        continue
                    elif mode.name == MODE.PARA_POSTPROC.name:
                        print("TODO: post-processing")
                    else:
                        pass
