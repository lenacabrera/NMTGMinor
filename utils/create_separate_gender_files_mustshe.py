import numpy as np
import json
import sys
import os
import argparse
import re

parser = argparse.ArgumentParser(description='create_separate_gender_files_mustshe.py')
parser.add_argument('-raw_path', required=True, default=None)


def main():

    opt = parser.parse_args()
    raw_path = opt.raw_path

    for ref in ["correct_ref", "wrong_ref"]:
        for gender in ["feminine", "masculine"]:
            for f in os.listdir(f"{raw_path}/{ref}"):
                if f.endswith(".s"):
                    lset = os.path.basename(f)[:5]
                    sl = lset.split("-")[0]
                    tl = lset.split("-")[1]

                    src_in = open(f"{raw_path}/{ref}/{lset}.s", "r", encoding="utf-8")
                    tgt_in = open(f"{raw_path}/{ref}/{lset}.t", "r", encoding="utf-8")

                    src_out = open(f"{raw_path}/{ref}/{gender}/{lset}.s", "w", encoding="utf-8")     
                    tgt_out = open(f"{raw_path}/{ref}/{gender}/{lset}.t", "w", encoding="utf-8")     

                    if tl != "en":
                        g_lan = tl
                    else:
                        g_lan = sl

                    category_in = open(f"{raw_path}/{ref}/{g_lan}_category.csv", "r", encoding="utf-8")
                    speaker_in = open(f"{raw_path}/{ref}/{g_lan}_speaker.csv", "r", encoding="utf-8")
                    gterms_in = open(f"{raw_path}/{ref}/{g_lan}_gterms.csv", "r", encoding="utf-8")

                    category_out = open(f"{raw_path}/{ref}/{gender}/annotation/{g_lan}_category.csv", "w", encoding="utf-8")
                    speaker_out = open(f"{raw_path}/{ref}/{gender}/annotation/{g_lan}_speaker.csv", "w", encoding="utf-8")
                    gterms_out = open(f"{raw_path}/{ref}/{gender}/annotation/{g_lan}_gterms.csv", "w", encoding="utf-8")

                    for src, tgt, ctg, spk, gtm in zip(src_in, tgt_in, category_in, speaker_in, gterms_in):
                        gender_word_forms = ctg[1]
                        if (gender_word_forms == "F" and gender == "feminine") or (gender_word_forms == "M" and gender == "masculine"):
                                src_out.write(src)
                                tgt_out.write(tgt)
                                category_out.write(ctg)
                                speaker_out.write(spk)
                                gterms_out.write(gtm)
                        else:
                            continue


def main_equal_f_m_instances():
    opt = parser.parse_args()
    raw_path = opt.raw_path

    for ref in ["correct_ref", "wrong_ref"]:
        for f in os.listdir(f"{raw_path}/{ref}"):
            if f.endswith(".s"):
                lset = os.path.basename(f)[:5]
                sl = lset.split("-")[0]
                tl = lset.split("-")[1]

                src_in = open(f"{raw_path}/{ref}/{lset}.s", "r", encoding="utf-8")
                tgt_in = open(f"{raw_path}/{ref}/{lset}.t", "r", encoding="utf-8")

                f_src_out = open(f"{raw_path}/{ref}/feminine/{lset}.s", "w", encoding="utf-8")     
                f_tgt_out = open(f"{raw_path}/{ref}/feminine/{lset}.t", "w", encoding="utf-8")     
                m_src_out = open(f"{raw_path}/{ref}/masculine/{lset}.s", "w", encoding="utf-8")     
                m_tgt_out = open(f"{raw_path}/{ref}/masculine/{lset}.t", "w", encoding="utf-8")     
                all_src_out = open(f"{raw_path}/{ref}/all/{lset}.s", "w", encoding="utf-8")     
                all_tgt_out = open(f"{raw_path}/{ref}/all/{lset}.t", "w", encoding="utf-8")     

                if tl != "en":
                    g_lan = tl
                else:
                    g_lan = sl

                category_in = open(f"{raw_path}/{ref}/{g_lan}_category.csv", "r", encoding="utf-8")
                speaker_in = open(f"{raw_path}/{ref}/{g_lan}_speaker.csv", "r", encoding="utf-8")
                gterms_in = open(f"{raw_path}/{ref}/{g_lan}_gterms.csv", "r", encoding="utf-8")

                f_category_out = open(f"{raw_path}/{ref}/feminine/annotation/{g_lan}_category.csv", "w", encoding="utf-8")
                f_speaker_out = open(f"{raw_path}/{ref}/feminine/annotation/{g_lan}_speaker.csv", "w", encoding="utf-8")
                f_gterms_out = open(f"{raw_path}/{ref}/feminine/annotation/{g_lan}_gterms.csv", "w", encoding="utf-8")

                m_category_out = open(f"{raw_path}/{ref}/masculine/annotation/{g_lan}_category.csv", "w", encoding="utf-8")
                m_speaker_out = open(f"{raw_path}/{ref}/masculine/annotation/{g_lan}_speaker.csv", "w", encoding="utf-8")
                m_gterms_out = open(f"{raw_path}/{ref}/masculine/annotation/{g_lan}_gterms.csv", "w", encoding="utf-8")
                
                all_category_out = open(f"{raw_path}/{ref}/all/annotation/{g_lan}_category.csv", "w", encoding="utf-8")
                all_speaker_out = open(f"{raw_path}/{ref}/all/annotation/{g_lan}_speaker.csv", "w", encoding="utf-8")
                all_gterms_out = open(f"{raw_path}/{ref}/all/annotation/{g_lan}_gterms.csv", "w", encoding="utf-8")

                f = {
                    "src": [],
                    "tgt": [],
                    "ctg": [],
                    "spk": [],
                    "gtm": []
                }

                m = {
                    "src": [],
                    "tgt": [],
                    "ctg": [],
                    "spk": [],
                    "gtm": []
                }

                for src, tgt, ctg, spk, gtm in zip(src_in, tgt_in, category_in, speaker_in, gterms_in):
                    gender_word_forms = ctg[1]
                    if gender_word_forms == "F":
                        f["src"].append(src)
                        f["tgt"].append(tgt)
                        f["ctg"].append(ctg)
                        f["spk"].append(spk)
                        f["gtm"].append(gtm)
                    if gender_word_forms == "M":
                        m["src"].append(src)
                        m["tgt"].append(tgt)
                        m["ctg"].append(ctg)
                        m["spk"].append(spk)
                        m["gtm"].append(gtm)

                equal_num_instances = min(len(f["src"]), len(m["src"]))
                n = equal_num_instances
                for src, tgt, ctg, spk, gtm in zip(f["src"][:n], f["tgt"][:n], f["ctg"][:n], f["spk"][:n], f["gtm"][:n]):
                    f_src_out.write(src)
                    f_tgt_out.write(tgt)
                    f_category_out.write(ctg)
                    f_speaker_out.write(spk)
                    f_gterms_out.write(gtm)

                for src, tgt, ctg, spk, gtm in zip(m["src"][:n], m["tgt"][:n], m["ctg"][:n], m["spk"][:n], m["gtm"][:n]):
                    m_src_out.write(src)
                    m_tgt_out.write(tgt)
                    m_category_out.write(ctg)
                    m_speaker_out.write(spk)
                    m_gterms_out.write(gtm)

                all = {
                    "src": f["src"][:n] + m["src"][:n],
                    "tgt": f["tgt"][:n] + m["tgt"][:n],
                    "ctg": f["ctg"][:n] + m["ctg"][:n],
                    "spk": f["spk"][:n] + m["spk"][:n],
                    "gtm": f["gtm"][:n] + m["gtm"][:n]
                }
                print(len(all["src"]))
                print(len(all["tgt"]))
                print(len(all["ctg"]))
                print(len(all["spk"]))
                print(len(all["gtm"]))
                print()
                i = 0
                for src, tgt, ctg, spk, gtm in zip(all["src"], all["tgt"], all["ctg"], all["spk"], all["gtm"]):
                    all_src_out.write(src)
                    all_tgt_out.write(tgt)
                    all_category_out.write(ctg)
                    all_speaker_out.write(spk)
                    all_gterms_out.write(gtm)
                    i += 1
                # print(i)


if __name__ == '__main__':
    # main()
    main_equal_f_m_instances()
