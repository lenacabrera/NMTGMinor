from unittest import result
import numpy as np
import json
import sys
import os
import argparse
import re
import math
import pandas as pd
import pickle
import os

from sqlalchemy import true

parser = argparse.ArgumentParser(description='eval_gender_accuracy.py')

parser.add_argument('-raw_path', required=True, default=None)
parser.add_argument('-pred_path', required=True, default=None)
# parser.add_argument('-pivot_language', required=True, default=None)
parser.add_argument('-train_set', required=True, default=None)
parser.add_argument('-out_path', required=True, default=None)


def get_bleu_scores_mustshe(lines):
    bleu_scores = []
    for line in lines:
        if line.split()[0] == "\"score\":":
            bleu_score = re.search(r"[0-9]*.[0-9]*", line.split()[1]).group(0)
            bleu_scores.append(float(bleu_score))
    avg_bleu = round(np.average(np.array(bleu_scores)), 1)
    return avg_bleu


def get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl):

    l = tl
    if tl == "en":
        l = sl

    gterms_file = open(f"{raw_path}/{ref}/{gender_set}/annotation/{l}_gterms.csv", "r", encoding="utf-8")
    speaker_file = open(f"{raw_path}/{ref}/{gender_set}/annotation/{l}_speaker.csv", "r", encoding="utf-8")
    category_file = open(f"{raw_path}/{ref}/{gender_set}/annotation/{l}_category.csv", "r", encoding="utf-8")  

    if pl == None:
        pred_file = open(f"{pred_path}/{ref}/{gender_set}/{f}", "r", encoding="utf-8")
    else:
        pred_file = open(f"{pred_path}/pivot/{ref}/{gender_set}/{sl}-{pl}-{tl}.real.pivotout.t.pt", "r", encoding="utf-8")

    accuracies_total = []
    accuracies_1 = []
    accuracies_2 = []
    accuracies_f_speaker = []
    accuracies_m_speaker = []

    for pred, gterms, speaker, category in zip(pred_file, gterms_file, speaker_file, category_file):
        pred_gterms = []
        gterms_list = [t for t in gterms.split(" ") if (t != '' and t != '\n')]
        for gterm in gterms_list:
            if gterm in pred:
                pred_gterms.append(gterm)
        # speaker gender
        acc = len(pred_gterms) / len(gterms_list)
        accuracies_total.append(acc)
        if speaker.replace("\n", "").lower() == "she":
            accuracies_f_speaker.append(acc)
        if speaker.replace("\n", "").lower() == "he":
            accuracies_m_speaker.append(acc)
        # gender of referred entity
        if "1" in category.replace("\n", ""):
            accuracies_1.append(acc)
        if "2" in category.replace("\n", ""):
            accuracies_2.append(acc)

    avg_acc_total = round(np.average(np.array(accuracies_total)) * 100, 1)
    avg_acc_f_speaker = round(np.average(np.array(accuracies_f_speaker)) * 100, 1)
    avg_acc_m_speaker = round(np.average(np.array(accuracies_m_speaker)) * 100, 1)

    avg_acc_1 = round(np.average(np.array(accuracies_1)) * 100, 1)
    avg_acc_2 = round(np.average(np.array(accuracies_2)) * 100, 1)

    return avg_acc_total, avg_acc_1, avg_acc_2, avg_acc_f_speaker, avg_acc_m_speaker

def count_num_of_instances():
    # num_all = len(accuracies_f) + len(accuracies_m)
    # num_f = len(accuracies_f)
    # num_m = len(accuracies_m)
    # num_1F = len(accuracies_1F)
    # num_2F = len(accuracies_2F)
    # num_1M = len(accuracies_1M)
    # num_2M = len(accuracies_2M)
    # return num_all, num_f, num_m, num_1F, num_2F, num_1M, num_2M
    raise NotImplementedError

def get_empty_results_dict_old():
    results = {
        "BLEU": {
            "zero_shot": {
                "correct_ref": {
                    "all": {},
                    "feminine": {},
                    "masculine": {}
                },
                "wrong_ref": {
                    "all": {},
                    "feminine": {},
                    "masculine": {}
                }
            },
            "pivot": {
                "correct_ref": {
                    "all": {},
                    "feminine": {},
                    "masculine": {}
                },
                "wrong_ref": {
                    "all": {},
                    "feminine": {},
                    "masculine": {}
                }
            }
        },
        "accuracy": {
            "zero_shot": {
                "correct_ref": {
                    "all": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "feminine": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "masculine": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    }
                },
                "wrong_ref": {
                    "all": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "feminine": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "masculine": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    }
                }
            },
            "pivot": {
                "correct_ref": {
                    "all": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "feminine": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "masculine": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    }
                },
                "wrong_ref": {
                    "all": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "feminine": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "masculine": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    }
                }
            }
        }
    }

    return results

def get_empty_results_dict():
    results = {
        "BLEU": {
            "zero_shot": {
                "all": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_bleu_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "feminine": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_bleu_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "masculine": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_bleu_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "bleu_f_of_all_c": {},
                "bleu_m_of_all_c": {},
                "diff_bleu_f_m_of_all_c": {},
                "tquality_w_gender_performance": {},
            },
            "pivot": {
                "all": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_bleu_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "feminine": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_bleu_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "masculine": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_bleu_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "bleu_f_of_all_c": {},
                "bleu_m_of_all_c": {},
                "diff_bleu_f_m_of_all_c": {},
                "tquality_w_gender_performance": {},
            }
        },
        "accuracy": {
            "zero_shot": {
                "all": {
                    "correct_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "wrong_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "diff_acc_c_w": {},
                    "sum_c_and_diff_c_w": {},
                    "cat_1_diff_c_w": {},
                    "cat_2_diff_c_w": {},
                    "cat_1_sum_c_and_diff_c_w": {},
                    "cat_2_sum_c_and_diff_c_w": {},
                    "fspeaker_diff_c_w": {},
                    "mspeaker_diff_c_w": {},
                    "fspeaker_sum_c_and_diff_c_w": {},
                    "mspeaker_sum_c_and_diff_c_w": {}
                },
                "feminine": {
                    "correct_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "wrong_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "diff_acc_c_w": {},
                    "sum_c_and_diff_c_w": {},
                    "cat_1_diff_c_w": {},
                    "cat_2_diff_c_w": {},
                    "cat_1_sum_c_and_diff_c_w": {},
                    "cat_2_sum_c_and_diff_c_w": {},
                    "fspeaker_diff_c_w": {},
                    "mspeaker_diff_c_w": {},
                    "fspeaker_sum_c_and_diff_c_w": {},
                    "mspeaker_sum_c_and_diff_c_w": {}
                },
                "masculine": {
                    "correct_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "wrong_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "diff_acc_c_w": {},
                    "sum_c_and_diff_c_w": {},
                    "cat_1_diff_c_w": {},
                    "cat_2_diff_c_w": {},
                    "cat_1_sum_c_and_diff_c_w": {},
                    "cat_2_sum_c_and_diff_c_w": {},
                    "fspeaker_diff_c_w": {},
                    "mspeaker_diff_c_w": {},
                    "fspeaker_sum_c_and_diff_c_w": {},
                    "mspeaker_sum_c_and_diff_c_w": {}
                },
                "acc_f_of_all_c": {},
                "acc_m_of_all_c": {},
                "diff_acc_f_m_of_all_c": {},
                "tquality_w_gender_performance": {},
                "cat_1_f_of_all_c": {},
                "cat_2_f_of_all_c": {},
                "cat_1_m_of_all_c": {},
                "cat_2_m_of_all_c": {},
                "cat_1_diff_f_m_of_all_c": {},
                "cat_2_diff_f_m_of_all_c": {},
                "cat_1_tquality_w_gender_performance": {},
                "cat_2_tquality_w_gender_performance": {},
                "fspeaker_f_of_all_c": {},
                "mspeaker_f_of_all_c": {},
                "fspeaker_m_of_all_c": {},
                "mspeaker_m_of_all_c": {},
                "fspeaker_diff_f_m_of_all_c": {},
                "mspeaker_diff_f_m_of_all_c": {},
                "fspeaker_tquality_w_gender_performance": {},
                "mspeaker_tquality_w_gender_performance": {}
            },
            "pivot": {
                "all": {
                    "correct_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "wrong_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "diff_acc_c_w": {},
                    "sum_c_and_diff_c_w": {},
                    "cat_1_diff_c_w": {},
                    "cat_2_diff_c_w": {},
                    "cat_1_sum_c_and_diff_c_w": {},
                    "cat_2_sum_c_and_diff_c_w": {},
                    "fspeaker_diff_c_w": {},
                    "mspeaker_diff_c_w": {},
                    "fspeaker_sum_c_and_diff_c_w": {},
                    "mspeaker_sum_c_and_diff_c_w": {}
                },
                "feminine": {
                    "correct_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "wrong_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "diff_acc_c_w": {},
                    "sum_c_and_diff_c_w": {},
                    "cat_1_diff_c_w": {},
                    "cat_2_diff_c_w": {},
                    "cat_1_sum_c_and_diff_c_w": {},
                    "cat_2_sum_c_and_diff_c_w": {},
                    "fspeaker_diff_c_w": {},
                    "mspeaker_diff_c_w": {},
                    "fspeaker_sum_c_and_diff_c_w": {},
                    "mspeaker_sum_c_and_diff_c_w": {}
                },
                "masculine": {
                    "correct_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "wrong_ref": {
                        "total": {},
                        "1": {},
                        "2": {},
                        "female_speaker": {},
                        "male_speaker": {}
                    },
                    "diff_acc_c_w": {},
                    "sum_c_and_diff_c_w": {},
                    "cat_1_diff_c_w": {},
                    "cat_2_diff_c_w": {},
                    "cat_1_sum_c_and_diff_c_w": {},
                    "cat_2_sum_c_and_diff_c_w": {},
                    "fspeaker_diff_c_w": {},
                    "mspeaker_diff_c_w": {},
                    "fspeaker_sum_c_and_diff_c_w": {},
                    "mspeaker_sum_c_and_diff_c_w": {}
                },
                "acc_f_of_all_c": {},
                "acc_m_of_all_c": {},
                "diff_acc_f_m_of_all_c": {},
                "tquality_w_gender_performance": {},
                "cat_1_f_of_all_c": {},
                "cat_2_f_of_all_c": {},
                "cat_1_m_of_all_c": {},
                "cat_2_m_of_all_c": {},
                "cat_1_diff_f_m_of_all_c": {},
                "cat_2_diff_f_m_of_all_c": {},
                "cat_1_tquality_w_gender_performance": {},
                "cat_2_tquality_w_gender_performance": {},
                "fspeaker_f_of_all_c": {},
                "mspeaker_f_of_all_c": {},
                "fspeaker_m_of_all_c": {},
                "mspeaker_m_of_all_c": {},
                "fspeaker_diff_f_m_of_all_c": {},
                "mspeaker_diff_f_m_of_all_c": {},
                "fspeaker_tquality_w_gender_performance": {},
                "mspeaker_tquality_w_gender_performance": {}
            }
        }   
    }

    return results

def export_bleu(results, out_path, train_set, map_train_set_model_name):
    out_file = f"{out_path}/summary_bleu.csv"
    if os.path.exists(out_file):
        df = pd.read_csv(out_file, sep=";")
    else:
        with open("/home/lperez/output/df_bleu.pkl", "rb") as file:
            df = pickle.load(file)

    sl = ''
    tl = ''
    cur_model = ""
    for i, r in df.iterrows():
        if i < 2:
            continue
        else:
            if sl == '' or not pd.isna(df.loc[i, "sl"]):
                sl = df.loc[i, "sl"]
            if tl == '' or not pd.isna(df.loc[i, "tl"]):
                tl = r["tl"]
            if cur_model == "" or not pd.isna(df.loc[i, "model"]):
                cur_model = r["model"]

            if map_train_set_model_name[train_set] == cur_model:

                if f"{sl}-{tl}" in results["BLEU"]["zero_shot"]["all"]["correct_ref"]:
                    # all, correct
                    df.loc[i, "all_cor_zs"] = results["BLEU"]["zero_shot"]["all"]["correct_ref"][f"{sl}-{tl}"]
                    # all, wrong 
                    df.loc[i, "all_wro_zs"] = results["BLEU"]["zero_shot"]["all"]["wrong_ref"][f"{sl}-{tl}"]
                    # all, diff
                    df.loc[i, "all_diff_zs"] = results["BLEU"]["zero_shot"]["all"]["diff_bleu_c_w"][f"{sl}-{tl}"]
                    # all, bleu + diff
                    df.loc[i, "all_sum_bleu_diff_zs"] = results["BLEU"]["zero_shot"]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # feminine, correct
                    df.loc[i, "f_cor_zs"] = results["BLEU"]["zero_shot"]["feminine"]["correct_ref"][f"{sl}-{tl}"]
                    # feminine, wrong
                    df.loc[i, "f_wro_zs"] = results["BLEU"]["zero_shot"]["feminine"]["wrong_ref"][f"{sl}-{tl}"]
                    # feminine, diff
                    df.loc[i, "f_diff_zs"] = results["BLEU"]["zero_shot"]["feminine"]["diff_bleu_c_w"][f"{sl}-{tl}"]
                    # feminine, bleu + diff
                    df.loc[i, "f_sum_bleu_diff_zs"] = results["BLEU"]["zero_shot"]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # masculine, correct
                    df.loc[i, "m_cor_zs"] = results["BLEU"]["zero_shot"]["masculine"]["correct_ref"][f"{sl}-{tl}"]
                    # masculine, wrong
                    df.loc[i, "m_wro_zs"] = results["BLEU"]["zero_shot"]["masculine"]["wrong_ref"][f"{sl}-{tl}"]
                    # masculine, diff
                    df.loc[i, "m_diff_zs"] = results["BLEU"]["zero_shot"]["masculine"]["diff_bleu_c_w"][f"{sl}-{tl}"]
                    # masculine, bleu + diff
                    df.loc[i, "m_sum_bleu_diff_zs"] = results["BLEU"]["zero_shot"]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # additional metrics
                    df.loc[i, "f_of_cor_zs"] = results["BLEU"]["zero_shot"]["bleu_f_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "m_of_cor_zs"] = results["BLEU"]["zero_shot"]["bleu_m_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "diff_f_m_of_cor_zs"] = results["BLEU"]["zero_shot"]["diff_bleu_f_m_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "gatq_zs"] = results["BLEU"]["zero_shot"]["tquality_w_gender_performance"][f"{sl}-{tl}"]

                if f"{sl}-{tl}" in results["BLEU"]["pivot"]["all"]["correct_ref"]:
                    # all, correct
                    df.loc[i, "all_cor_pv"] = results["BLEU"]["pivot"]["all"]["correct_ref"][f"{sl}-{tl}"]
                    # all, wrong 
                    df.loc[i, "all_wro_pv"] = results["BLEU"]["pivot"]["all"]["wrong_ref"][f"{sl}-{tl}"]
                    # all, diff
                    df.loc[i, "all_diff_pv"] = results["BLEU"]["pivot"]["all"]["diff_bleu_c_w"][f"{sl}-{tl}"]
                    # all, bleu + diff
                    df.loc[i, "all_sum_bleu_diff_pv"] = results["BLEU"]["pivot"]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                   # feminine, correct
                    df.loc[i, "f_cor_pv"] = results["BLEU"]["pivot"]["feminine"]["correct_ref"][f"{sl}-{tl}"]
                    # feminine, wrong
                    df.loc[i, "f_wro_pv"] = results["BLEU"]["pivot"]["feminine"]["wrong_ref"][f"{sl}-{tl}"]
                    # feminine, diff
                    df.loc[i, "f_diff_pv"] = results["BLEU"]["pivot"]["feminine"]["diff_bleu_c_w"][f"{sl}-{tl}"]
                    # feminine, bleu + diff
                    df.loc[i, "f_sum_bleu_diff_pv"] = results["BLEU"]["pivot"]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # masculine, correct
                    df.loc[i, "m_cor_pv"] = results["BLEU"]["pivot"]["masculine"]["correct_ref"][f"{sl}-{tl}"]
                    # masculine, wrong
                    df.loc[i, "m_wro_pv"] = results["BLEU"]["pivot"]["masculine"]["wrong_ref"][f"{sl}-{tl}"]
                    # masculine, diff
                    df.loc[i, "m_diff_pv"] = results["BLEU"]["pivot"]["masculine"]["diff_bleu_c_w"][f"{sl}-{tl}"]
                    # masculine, bleu + diff
                    df.loc[i, "m_sum_bleu_diff_pv"] = results["BLEU"]["pivot"]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # additional metrics
                    df.loc[i, "f_of_cor_pv"] = results["BLEU"]["pivot"]["bleu_f_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "m_of_cor_pv"] = results["BLEU"]["pivot"]["bleu_m_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "diff_f_m_of_cor_pv"] = results["BLEU"]["pivot"]["diff_bleu_f_m_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "gatq_pv"] = results["BLEU"]["pivot"]["tquality_w_gender_performance"][f"{sl}-{tl}"]

    df.to_csv(out_file, index=False, sep=";")


def export_acc(results, out_path, train_set, map_train_set_model_name):

    out_file = f"{out_path}/summary_acc.csv"
    if os.path.exists(out_file):
        df = pd.read_csv(out_file, sep=";")
    else:
        with open("/home/lperez/output/df_acc.pkl", "rb") as file:
            df = pickle.load(file)

    sl = ''
    tl = ''
    cur_model = ""
    for i, r in df.iterrows():
        if i < 2:
            continue
        else:
            if sl == '' or not pd.isna(df.loc[i, "sl"]):
                sl = df.loc[i, "sl"]
            if tl == '' or not pd.isna(df.loc[i, "tl"]):
                tl = r["tl"]
            if cur_model == "" or not pd.isna(df.loc[i, "model"]):
                cur_model = r["model"]

            if map_train_set_model_name[train_set] == cur_model:

                if f"{sl}-{tl}" in results["accuracy"]["zero_shot"]["all"]["correct_ref"]["total"]:
                    # all, correct
                    df.loc[i, "all_cor_zs"] = results["accuracy"]["zero_shot"]["all"]["correct_ref"]["total"][f"{sl}-{tl}"]
                    # all, wrong 
                    df.loc[i, "all_wro_zs"] = results["accuracy"]["zero_shot"]["all"]["wrong_ref"]["total"][f"{sl}-{tl}"]
                    # all, diff
                    df.loc[i, "all_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["diff_acc_c_w"][f"{sl}-{tl}"]
                    # all, bleu + diff
                    df.loc[i, "all_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # feminine, correct
                    df.loc[i, "f_cor_zs"] = results["accuracy"]["zero_shot"]["feminine"]["correct_ref"]["total"][f"{sl}-{tl}"]
                    # feminine, wrong
                    df.loc[i, "f_wro_zs"] = results["accuracy"]["zero_shot"]["feminine"]["wrong_ref"]["total"][f"{sl}-{tl}"]
                    # feminine, diff
                    df.loc[i, "f_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["diff_acc_c_w"][f"{sl}-{tl}"]
                    # feminine, bleu + diff
                    df.loc[i, "f_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # masculine, correct
                    df.loc[i, "m_cor_zs"] = results["accuracy"]["zero_shot"]["masculine"]["correct_ref"]["total"][f"{sl}-{tl}"]
                    # masculine, wrong
                    df.loc[i, "m_wro_zs"] = results["accuracy"]["zero_shot"]["masculine"]["wrong_ref"]["total"][f"{sl}-{tl}"]
                    # masculine, diff
                    df.loc[i, "m_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["diff_acc_c_w"][f"{sl}-{tl}"]
                    # masculine, bleu + diff
                    df.loc[i, "m_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # additional metrics
                    df.loc[i, "f_of_cor_zs"] = results["accuracy"]["zero_shot"]["acc_f_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "m_of_cor_zs"] = results["accuracy"]["zero_shot"]["acc_m_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "diff_f_m_of_cor_zs"] = results["accuracy"]["zero_shot"]["diff_acc_f_m_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "gatq_zs"] = results["accuracy"]["zero_shot"]["tquality_w_gender_performance"][f"{sl}-{tl}"]

                if f"{sl}-{tl}" in results["accuracy"]["pivot"]["all"]["correct_ref"]["total"]:
                    # all, correct
                    df.loc[i, "all_cor_pv"] = results["accuracy"]["pivot"]["all"]["correct_ref"]["total"][f"{sl}-{tl}"]
                    # all, wrong 
                    df.loc[i, "all_wro_pv"] = results["accuracy"]["pivot"]["all"]["wrong_ref"]["total"][f"{sl}-{tl}"]
                    # all, diff
                    df.loc[i, "all_diff_pv"] = results["accuracy"]["pivot"]["all"]["diff_acc_c_w"][f"{sl}-{tl}"]
                    # all, bleu + diff
                    df.loc[i, "all_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                   # feminine, correct
                    df.loc[i, "f_cor_pv"] = results["accuracy"]["pivot"]["feminine"]["correct_ref"]["total"][f"{sl}-{tl}"]
                    # feminine, wrong
                    df.loc[i, "f_wro_pv"] = results["accuracy"]["pivot"]["feminine"]["wrong_ref"]["total"][f"{sl}-{tl}"]
                    # feminine, diff
                    df.loc[i, "f_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["diff_acc_c_w"][f"{sl}-{tl}"]
                    # feminine, bleu + diff
                    df.loc[i, "f_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # masculine, correct
                    df.loc[i, "m_cor_pv"] = results["accuracy"]["pivot"]["masculine"]["correct_ref"]["total"][f"{sl}-{tl}"]
                    # masculine, wrong
                    df.loc[i, "m_wro_pv"] = results["accuracy"]["pivot"]["masculine"]["wrong_ref"]["total"][f"{sl}-{tl}"]
                    # masculine, diff
                    df.loc[i, "m_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["diff_acc_c_w"][f"{sl}-{tl}"]
                    # masculine, bleu + diff
                    df.loc[i, "m_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # additional metrics
                    df.loc[i, "f_of_cor_pv"] = results["accuracy"]["pivot"]["acc_f_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "m_of_cor_pv"] = results["accuracy"]["pivot"]["acc_m_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "diff_f_m_of_cor_pv"] = results["accuracy"]["pivot"]["diff_acc_f_m_of_all_c"][f"{sl}-{tl}"]
                    df.loc[i, "gatq_pv"] = results["accuracy"]["pivot"]["tquality_w_gender_performance"][f"{sl}-{tl}"]

    df.to_csv(out_file, index=False, sep=";")

def export_acc_cat(results, out_path, train_set, map_train_set_model_name):

    out_file = f"{out_path}/summary_acc_cat.csv"
    if os.path.exists(out_file):
        df = pd.read_csv(out_file, sep=";")
    else:
        with open("/home/lperez/output/df_acc_cat.pkl", "rb") as file:
            df = pickle.load(file)

    sl = ''
    tl = ''
    cur_model = ""
    for i, r in df.iterrows():
        if i < 2:
            continue
        else:
            if i % 2 == 0:
                if sl == '' or not pd.isna(df.loc[i, "sl"]):
                    sl = df.loc[i, "sl"]
                if tl == '' or not pd.isna(df.loc[i, "tl"]):
                    tl = r["tl"]
                if cur_model == "" or not pd.isna(df.loc[i, "model"]):
                    cur_model = r["model"]

                if map_train_set_model_name[train_set] == cur_model:

                    if f"{sl}-{tl}" in results["accuracy"]["zero_shot"]["all"]["correct_ref"]["1"]:
                        # all, correct
                        df.loc[i, "all_cor_zs"] = results["accuracy"]["zero_shot"]["all"]["correct_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_cor_zs"] = results["accuracy"]["zero_shot"]["all"]["correct_ref"]["2"][f"{sl}-{tl}"]
                        # all, wrong 
                        df.loc[i, "all_wro_zs"] = results["accuracy"]["zero_shot"]["all"]["wrong_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_wro_zs"] = results["accuracy"]["zero_shot"]["all"]["wrong_ref"]["2"][f"{sl}-{tl}"]
                        # all, diff
                        df.loc[i, "all_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["cat_1_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["cat_2_diff_c_w"][f"{sl}-{tl}"]
                        # all, bleu + diff
                        df.loc[i, "all_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["cat_1_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["cat_2_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # feminine, correct
                        df.loc[i, "f_cor_zs"] = results["accuracy"]["zero_shot"]["feminine"]["correct_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_cor_zs"] = results["accuracy"]["zero_shot"]["feminine"]["correct_ref"]["2"][f"{sl}-{tl}"]
                        # feminine, wrong
                        df.loc[i, "f_wro_zs"] = results["accuracy"]["zero_shot"]["feminine"]["wrong_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_wro_zs"] = results["accuracy"]["zero_shot"]["feminine"]["wrong_ref"]["2"][f"{sl}-{tl}"]
                        # feminine, diff
                        df.loc[i, "f_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["cat_1_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["cat_2_diff_c_w"][f"{sl}-{tl}"]
                        # feminine, bleu + diff
                        df.loc[i, "f_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["cat_1_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["cat_2_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # masculine, correct
                        df.loc[i, "m_cor_zs"] = results["accuracy"]["zero_shot"]["masculine"]["correct_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_cor_zs"] = results["accuracy"]["zero_shot"]["masculine"]["correct_ref"]["2"][f"{sl}-{tl}"]
                        # masculine, wrong
                        df.loc[i, "m_wro_zs"] = results["accuracy"]["zero_shot"]["masculine"]["wrong_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_wro_zs"] = results["accuracy"]["zero_shot"]["masculine"]["wrong_ref"]["2"][f"{sl}-{tl}"]
                        # masculine, diff
                        df.loc[i, "m_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["cat_1_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["cat_2_diff_c_w"][f"{sl}-{tl}"]
                        # masculine, bleu + diff
                        df.loc[i, "m_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["cat_1_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["cat_2_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # additional metrics
                        df.loc[i, "f_of_cor_zs"] = results["accuracy"]["zero_shot"]["cat_1_f_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_of_cor_zs"] = results["accuracy"]["zero_shot"]["cat_2_f_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "m_of_cor_zs"] = results["accuracy"]["zero_shot"]["cat_1_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_of_cor_zs"] = results["accuracy"]["zero_shot"]["cat_2_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "diff_f_m_of_cor_zs"] = results["accuracy"]["zero_shot"]["cat_1_diff_f_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "diff_f_m_of_cor_zs"] = results["accuracy"]["zero_shot"]["cat_2_diff_f_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "gatq_zs"] = results["accuracy"]["zero_shot"]["cat_1_tquality_w_gender_performance"][f"{sl}-{tl}"]
                        df.loc[i + 1, "gatq_zs"] = results["accuracy"]["zero_shot"]["cat_2_tquality_w_gender_performance"][f"{sl}-{tl}"]

                    if f"{sl}-{tl}" in results["accuracy"]["pivot"]["all"]["correct_ref"]["1"]:
                        # all, correct
                        df.loc[i, "all_cor_pv"] = results["accuracy"]["pivot"]["all"]["correct_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_cor_pv"] = results["accuracy"]["pivot"]["all"]["correct_ref"]["2"][f"{sl}-{tl}"]
                        # all, wrong 
                        df.loc[i, "all_wro_pv"] = results["accuracy"]["pivot"]["all"]["wrong_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_wro_pv"] = results["accuracy"]["pivot"]["all"]["wrong_ref"]["2"][f"{sl}-{tl}"]
                        # all, diff
                        df.loc[i, "all_diff_pv"] = results["accuracy"]["pivot"]["all"]["cat_1_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_diff_pv"] = results["accuracy"]["pivot"]["all"]["cat_2_diff_c_w"][f"{sl}-{tl}"]
                        # all, bleu + diff
                        df.loc[i, "all_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["all"]["cat_1_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["all"]["cat_2_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # feminine, correct
                        df.loc[i, "f_cor_pv"] = results["accuracy"]["pivot"]["feminine"]["correct_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_cor_pv"] = results["accuracy"]["pivot"]["feminine"]["correct_ref"]["2"][f"{sl}-{tl}"]
                        # feminine, wrong
                        df.loc[i, "f_wro_pv"] = results["accuracy"]["pivot"]["feminine"]["wrong_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_wro_pv"] = results["accuracy"]["pivot"]["feminine"]["wrong_ref"]["2"][f"{sl}-{tl}"]
                        # feminine, diff
                        df.loc[i, "f_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["cat_1_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["cat_2_diff_c_w"][f"{sl}-{tl}"]
                        # feminine, bleu + diff
                        df.loc[i, "f_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["cat_1_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["cat_2_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # masculine, correct
                        df.loc[i, "m_cor_pv"] = results["accuracy"]["pivot"]["masculine"]["correct_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_cor_pv"] = results["accuracy"]["pivot"]["masculine"]["correct_ref"]["2"][f"{sl}-{tl}"]
                        # masculine, wrong
                        df.loc[i, "m_wro_pv"] = results["accuracy"]["pivot"]["masculine"]["wrong_ref"]["1"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_wro_pv"] = results["accuracy"]["pivot"]["masculine"]["wrong_ref"]["2"][f"{sl}-{tl}"]
                        # masculine, diff
                        df.loc[i, "m_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["cat_1_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["cat_2_diff_c_w"][f"{sl}-{tl}"]
                        # masculine, bleu + diff
                        df.loc[i, "m_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["cat_1_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["cat_2_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # additional metrics
                        df.loc[i, "f_of_cor_pv"] = results["accuracy"]["pivot"]["cat_1_f_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_of_cor_pv"] = results["accuracy"]["pivot"]["cat_2_f_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "m_of_cor_pv"] = results["accuracy"]["pivot"]["cat_1_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_of_cor_pv"] = results["accuracy"]["pivot"]["cat_2_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "diff_f_m_of_cor_pv"] = results["accuracy"]["pivot"]["cat_1_diff_f_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "diff_f_m_of_cor_pv"] = results["accuracy"]["pivot"]["cat_2_diff_f_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "gatq_pv"] = results["accuracy"]["pivot"]["cat_1_tquality_w_gender_performance"][f"{sl}-{tl}"]
                        df.loc[i + 1, "gatq_pv"] = results["accuracy"]["pivot"]["cat_2_tquality_w_gender_performance"][f"{sl}-{tl}"]

    df.to_csv(out_file, index=False, sep=";")

def export_acc_speaker(results, out_path, train_set, map_train_set_model_name):

    out_file = f"{out_path}/summary_acc_speaker.csv"
    if os.path.exists(out_file):
        df = pd.read_csv(out_file, sep=";")
    else:
        with open("/home/lperez/output/df_acc_speaker.pkl", "rb") as file:
            df = pickle.load(file)

    sl = ''
    tl = ''
    cur_model = ""
    for i, r in df.iterrows():
        if i < 2:
            continue
        else:
            if i % 2 == 0:
                if sl == '' or not pd.isna(df.loc[i, "sl"]):
                    sl = df.loc[i, "sl"]
                if tl == '' or not pd.isna(df.loc[i, "tl"]):
                    tl = r["tl"]
                if cur_model == "" or not pd.isna(df.loc[i, "model"]):
                    cur_model = r["model"]

                if map_train_set_model_name[train_set] == cur_model:

                    if f"{sl}-{tl}" in results["accuracy"]["zero_shot"]["all"]["correct_ref"]["female_speaker"]:
                        # all, correct
                        df.loc[i, "all_cor_zs"] = results["accuracy"]["zero_shot"]["all"]["correct_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_cor_zs"] = results["accuracy"]["zero_shot"]["all"]["correct_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # all, wrong 
                        df.loc[i, "all_wro_zs"] = results["accuracy"]["zero_shot"]["all"]["wrong_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_wro_zs"] = results["accuracy"]["zero_shot"]["all"]["wrong_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # all, diff
                        df.loc[i, "all_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["fspeaker_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["mspeaker_diff_c_w"][f"{sl}-{tl}"]
                        # all, bleu + diff
                        df.loc[i, "all_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["fspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["all"]["mspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # feminine, correct
                        df.loc[i, "f_cor_zs"] = results["accuracy"]["zero_shot"]["feminine"]["correct_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_cor_zs"] = results["accuracy"]["zero_shot"]["feminine"]["correct_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # feminine, wrong
                        df.loc[i, "f_wro_zs"] = results["accuracy"]["zero_shot"]["feminine"]["wrong_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_wro_zs"] = results["accuracy"]["zero_shot"]["feminine"]["wrong_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # feminine, diff
                        df.loc[i, "f_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["fspeaker_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["mspeaker_diff_c_w"][f"{sl}-{tl}"]
                        # feminine, bleu + diff
                        df.loc[i, "f_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["fspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["feminine"]["mspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # masculine, correct
                        df.loc[i, "m_cor_zs"] = results["accuracy"]["zero_shot"]["masculine"]["correct_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_cor_zs"] = results["accuracy"]["zero_shot"]["masculine"]["correct_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # masculine, wrong
                        df.loc[i, "m_wro_zs"] = results["accuracy"]["zero_shot"]["masculine"]["wrong_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_wro_zs"] = results["accuracy"]["zero_shot"]["masculine"]["wrong_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # masculine, diff
                        df.loc[i, "m_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["fspeaker_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["mspeaker_diff_c_w"][f"{sl}-{tl}"]
                        # masculine, bleu + diff
                        df.loc[i, "m_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["fspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_sum_bleu_diff_zs"] = results["accuracy"]["zero_shot"]["masculine"]["mspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # additional metrics
                        df.loc[i, "f_of_cor_zs"] = results["accuracy"]["zero_shot"]["fspeaker_f_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_of_cor_zs"] = results["accuracy"]["zero_shot"]["mspeaker_f_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "m_of_cor_zs"] = results["accuracy"]["zero_shot"]["fspeaker_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_of_cor_zs"] = results["accuracy"]["zero_shot"]["mspeaker_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "diff_f_m_of_cor_zs"] = results["accuracy"]["zero_shot"]["fspeaker_diff_f_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "diff_f_m_of_cor_zs"] = results["accuracy"]["zero_shot"]["mspeaker_diff_f_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "gatq_zs"] = results["accuracy"]["zero_shot"]["fspeaker_tquality_w_gender_performance"][f"{sl}-{tl}"]
                        df.loc[i + 1, "gatq_zs"] = results["accuracy"]["zero_shot"]["mspeaker_tquality_w_gender_performance"][f"{sl}-{tl}"]

                    if f"{sl}-{tl}" in results["accuracy"]["pivot"]["all"]["correct_ref"]["female_speaker"]:
                        # all, correct
                        df.loc[i, "all_cor_pv"] = results["accuracy"]["pivot"]["all"]["correct_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_cor_pv"] = results["accuracy"]["pivot"]["all"]["correct_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # all, wrong 
                        df.loc[i, "all_wro_pv"] = results["accuracy"]["pivot"]["all"]["wrong_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_wro_pv"] = results["accuracy"]["pivot"]["all"]["wrong_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # all, diff
                        df.loc[i, "all_diff_pv"] = results["accuracy"]["pivot"]["all"]["fspeaker_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_diff_pv"] = results["accuracy"]["pivot"]["all"]["mspeaker_diff_c_w"][f"{sl}-{tl}"]
                        # all, bleu + diff
                        df.loc[i, "all_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["all"]["fspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "all_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["all"]["mspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                    # feminine, correct
                        df.loc[i, "f_cor_pv"] = results["accuracy"]["pivot"]["feminine"]["correct_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_cor_pv"] = results["accuracy"]["pivot"]["feminine"]["correct_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # feminine, wrong
                        df.loc[i, "f_wro_pv"] = results["accuracy"]["pivot"]["feminine"]["wrong_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_wro_pv"] = results["accuracy"]["pivot"]["feminine"]["wrong_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # feminine, diff
                        df.loc[i, "f_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["fspeaker_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["mspeaker_diff_c_w"][f"{sl}-{tl}"]
                        # feminine, bleu + diff
                        df.loc[i, "f_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["fspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["feminine"]["mspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # masculine, correct
                        df.loc[i, "m_cor_pv"] = results["accuracy"]["pivot"]["masculine"]["correct_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_cor_pv"] = results["accuracy"]["pivot"]["masculine"]["correct_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # masculine, wrong
                        df.loc[i, "m_wro_pv"] = results["accuracy"]["pivot"]["masculine"]["wrong_ref"]["female_speaker"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_wro_pv"] = results["accuracy"]["pivot"]["masculine"]["wrong_ref"]["male_speaker"][f"{sl}-{tl}"]
                        # masculine, diff
                        df.loc[i, "m_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["fspeaker_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["mspeaker_diff_c_w"][f"{sl}-{tl}"]
                        # masculine, bleu + diff
                        df.loc[i, "m_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["fspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_sum_bleu_diff_pv"] = results["accuracy"]["pivot"]["masculine"]["mspeaker_sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                        # additional metrics
                        df.loc[i, "f_of_cor_pv"] = results["accuracy"]["pivot"]["fspeaker_f_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "f_of_cor_pv"] = results["accuracy"]["pivot"]["mspeaker_f_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "m_of_cor_pv"] = results["accuracy"]["pivot"]["fspeaker_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "m_of_cor_pv"] = results["accuracy"]["pivot"]["mspeaker_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "diff_f_m_of_cor_pv"] = results["accuracy"]["pivot"]["fspeaker_diff_f_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i + 1, "diff_f_m_of_cor_pv"] = results["accuracy"]["pivot"]["mspeaker_diff_f_m_of_all_c"][f"{sl}-{tl}"]
                        df.loc[i, "gatq_pv"] = results["accuracy"]["pivot"]["fspeaker_tquality_w_gender_performance"][f"{sl}-{tl}"]
                        df.loc[i + 1, "gatq_pv"] = results["accuracy"]["pivot"]["mspeaker_tquality_w_gender_performance"][f"{sl}-{tl}"]

    df.to_csv(out_file, index=False, sep=";")


def export_results(results, out_path, train_set):

    # (1) all
    with open(f"{out_path}/{train_set}.json", 'w') as file:
        file.write(json.dumps(results, indent=3)) # use `json.loads` to do the reverse

    map_train_set_model_name = {
        "twoway.r32.q": "baseline_EN",
        "twoway.r32.q.new": "residual_EN",
        "multiwayES": "baseline_ES",
        "multiwayES.r32.q": "residual_ES",
        "multiwayESFRIT": "baseline_ESFRIT",
        "multiwayESFRIT.r32.q.new": "residual_ESFRIT",
        "multiwayDE": "baseline_DE",
        "multiwayDE.r32.q": "residual_DE",
    }

    # (2) BLEU
    export_bleu(results, out_path, train_set, map_train_set_model_name)

    # (3) Accuracy
    export_acc(results, out_path, train_set, map_train_set_model_name)

    # (4) Accuracy (cat)
    export_acc_cat(results, out_path, train_set, map_train_set_model_name)

    # (5) Accuracy (speaker)
    export_acc_speaker(results, out_path, train_set, map_train_set_model_name)


def main_mustshe():

    opt = parser.parse_args()
    raw_path = opt.raw_path
    pred_path = opt.pred_path
    train_set = opt.train_set
    out_path = opt.out_path
    
    results = get_empty_results_dict()

    for translation in ["zero_shot", "pivot"]:
        lsets = []
        for gender_set in ["all", "feminine", "masculine"]:
            for ref in ["correct_ref", "wrong_ref"]:
                if translation == "zero_shot":
                    # zero-shot
                    for f in os.listdir(f"{pred_path}/{ref}/{gender_set}"):
                        if os.path.isfile(os.path.join(f"{pred_path}/{ref}/{gender_set}", f)):
                            lset = re.search(r"[a-z][a-z]-[a-z][a-z]", os.path.basename(f)).group(0)
                            lsets.append(lset)
                            sl = lset.split("-")[0]
                            tl = lset.split("-")[1]
                            if f.endswith(".res"):
                                # BLEU
                                lines_zs = open(f"{pred_path}/{ref}/{gender_set}/{f}").readlines() 
                                bleu_zs = get_bleu_scores_mustshe(lines_zs)
                                results["BLEU"][translation][gender_set][ref][lset] = float(bleu_zs)
                            elif f.startswith(lset) and f.endswith(".pt"):
                                # Accuracy
                                avg_acc_total_zs, avg_acc_1_zs, avg_acc_2_zs, avg_acc_f_zs, avg_acc_m_zs = get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl=None)
                                results["accuracy"][translation][gender_set][ref]["total"][lset] = avg_acc_total_zs
                                results["accuracy"][translation][gender_set][ref]["1"][lset] = avg_acc_1_zs
                                results["accuracy"][translation][gender_set][ref]["2"][lset] = avg_acc_2_zs
                                results["accuracy"][translation][gender_set][ref]["female_speaker"][lset] = avg_acc_f_zs
                                results["accuracy"][translation][gender_set][ref]["male_speaker"][lset] = avg_acc_m_zs
                            else:
                                continue
                else:
                    # pivot
                    for f in os.listdir(f"{pred_path}/pivot/{ref}/{gender_set}"):
                        if os.path.isfile(os.path.join(f"{pred_path}/pivot/{ref}/{gender_set}", f)):
                            lset = re.search(r"[a-z][a-z]-[a-z][a-z]-[a-z][a-z]", os.path.basename(f)).group(0)
                            sl = lset.split("-")[0]
                            pl = lset.split("-")[1]
                            tl = lset.split("-")[2]
                            lset = f"{sl}-{tl}"
                            lsets.append(lset)
                            if sl != pl and tl != pl:
                                if f.endswith(".res"):
                                    # BLEU
                                    lines_pv = open(f"{pred_path}/pivot/{ref}/{gender_set}/{sl}-{pl}-{tl}.real.pivotout.t.res").readlines()
                                    bleu_pv = get_bleu_scores_mustshe(lines_pv)
                                    results["BLEU"][translation][gender_set][ref][lset] = float(bleu_pv)
                                elif f.startswith(f"{sl}-{pl}-{tl}") and f.endswith(".pt"):
                                    # Accuracy
                                    avg_acc_total_pv, avg_acc_1_pv, avg_acc_2_pv, avg_acc_f_pv, avg_acc_m_pv = get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl)
                                    results["accuracy"][translation][gender_set][ref]["total"][lset] = avg_acc_total_pv
                                    results["accuracy"][translation][gender_set][ref]["1"][lset] = avg_acc_1_pv
                                    results["accuracy"][translation][gender_set][ref]["2"][lset] = avg_acc_2_pv
                                    results["accuracy"][translation][gender_set][ref]["female_speaker"][lset] = avg_acc_f_pv
                                    results["accuracy"][translation][gender_set][ref]["male_speaker"][lset] = avg_acc_m_pv
                                else:
                                    continue   
                            else:
                                continue

            # additional metrics (I)
            for lset in lsets:
                if translation == "pivot":
                    if lset not in results["BLEU"][translation][gender_set]["correct_ref"]:
                        continue
                ## I1. BLEU
                ### (I1a) BLEU correct - BLEU wrong
                diff_bleu_c_w = np.round(results["BLEU"][translation][gender_set]["correct_ref"][lset] - results["BLEU"][translation][gender_set]["wrong_ref"][lset], 1)
                results["BLEU"][translation][gender_set]["diff_bleu_c_w"][lset] = diff_bleu_c_w
                ### (I1b) = BLEU correct + (I1a)
                bleu_sum_c_and_diff_c_w = np.round(results["BLEU"][translation][gender_set]["correct_ref"][lset] + diff_bleu_c_w, 1)
                results["BLEU"][translation][gender_set]["sum_c_and_diff_c_w"][lset] = bleu_sum_c_and_diff_c_w
                ## I2. Accuracy
                ### (I2a) Accuracy correct - BLEU wrong
                diff_acc_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["total"][lset] - results["accuracy"][translation][gender_set]["wrong_ref"]["total"][lset], 1)
                results["accuracy"][translation][gender_set]["diff_acc_c_w"][lset] = diff_acc_c_w
                ### (I1b) = BLEU correct + (I1a)
                acc_sum_c_and_diff_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["total"][lset] + diff_acc_c_w, 1)
                results["accuracy"][translation][gender_set]["sum_c_and_diff_c_w"][lset] = acc_sum_c_and_diff_c_w
                ## I3. Accuracy (category)
                # -> cat. 1
                ### (I3a) Accuracy correct - BLEU wrong
                diff_acc_cat_1_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["1"][lset] - results["accuracy"][translation][gender_set]["wrong_ref"]["1"][lset], 1)
                results["accuracy"][translation][gender_set]["cat_1_diff_c_w"][lset] = diff_acc_cat_1_c_w
                ### (I3b) = BLEU correct + (I3a)
                acc_cat_1_sum_c_and_diff_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["total"][lset] + diff_acc_cat_1_c_w, 1)
                results["accuracy"][translation][gender_set]["cat_1_sum_c_and_diff_c_w"][lset] = acc_cat_1_sum_c_and_diff_c_w
                # -> cat. 2
                ### (I3a) Accuracy correct - BLEU wrong
                diff_acc_cat_2_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["2"][lset] - results["accuracy"][translation][gender_set]["wrong_ref"]["2"][lset], 1)
                results["accuracy"][translation][gender_set]["cat_2_diff_c_w"][lset] = diff_acc_cat_2_c_w
                ### (I3b) = BLEU correct + (I3a)
                acc_cat_2_sum_c_and_diff_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["total"][lset] + diff_acc_cat_2_c_w, 1)
                results["accuracy"][translation][gender_set]["cat_2_sum_c_and_diff_c_w"][lset] = acc_cat_2_sum_c_and_diff_c_w
                ## I4. Accuracy (speaker)
                # -> female
                ### (I4a) Accuracy correct - BLEU wrong
                diff_acc_fspeaker_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["female_speaker"][lset] - results["accuracy"][translation][gender_set]["wrong_ref"]["female_speaker"][lset], 1)
                results["accuracy"][translation][gender_set]["fspeaker_diff_c_w"][lset] = diff_acc_fspeaker_c_w
                ### (I4b) = BLEU correct + (I4a)
                acc_fspeaker_sum_c_and_diff_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["female_speaker"][lset] + diff_acc_fspeaker_c_w, 1)
                results["accuracy"][translation][gender_set]["fspeaker_sum_c_and_diff_c_w"][lset] = acc_fspeaker_sum_c_and_diff_c_w
                # -> male
                ### (I4a) Accuracy correct - BLEU wrong
                diff_acc_mspeaker_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["male_speaker"][lset] - results["accuracy"][translation][gender_set]["wrong_ref"]["male_speaker"][lset], 1)
                results["accuracy"][translation][gender_set]["mspeaker_diff_c_w"][lset] = diff_acc_mspeaker_c_w
                ### (I4b) = BLEU correct + (I4a)
                acc_mspeaker_sum_c_and_diff_c_w = np.round(results["accuracy"][translation][gender_set]["correct_ref"]["male_speaker"][lset] + diff_acc_mspeaker_c_w, 1)
                results["accuracy"][translation][gender_set]["mspeaker_sum_c_and_diff_c_w"][lset] = acc_mspeaker_sum_c_and_diff_c_w

        # additional metrics (II)
        for lset in lsets:
            ## II2. BLEU
            f_bleu_c = results["BLEU"][translation]["feminine"]["correct_ref"][lset]
            m_bleu_c = results["BLEU"][translation]["masculine"]["correct_ref"][lset]

            ### (II1c) proportion of feminine correct BLEU on sum of feminine and masculine correct BLEU 
            bleu_f_of_all_c = np.round((f_bleu_c / (f_bleu_c + m_bleu_c)) * 100, 1)
            results["BLEU"][translation]["bleu_f_of_all_c"][lset] = bleu_f_of_all_c
            ### (II1d) proportion of masculine correct BLEU on sum of feminine and masculine correct BLEU 
            bleu_m_of_all_c = np.round((m_bleu_c / (f_bleu_c + m_bleu_c)) * 100, 1)
            results["BLEU"][translation]["bleu_m_of_all_c"][lset] = bleu_m_of_all_c
            ### (II1e) = (II1c) - (II1d)
            diff_bleu_f_m_of_all_c = np.round(bleu_f_of_all_c - bleu_m_of_all_c, 1)
            results["BLEU"][translation]["diff_bleu_f_m_of_all_c"][lset] = diff_bleu_f_m_of_all_c

            f_bleu_sum_c_and_diff_c_w = results["BLEU"][translation]["feminine"]["sum_c_and_diff_c_w"][lset]
            m_bleu_sum_c_and_diff_c_w = results["BLEU"][translation]["masculine"]["sum_c_and_diff_c_w"][lset]
            ### (II1f) avg. of f(I1b) and m(I1b) - abs. diff. between f(I1b) and m(I1b)
            bleu_tquality_w_gender_performance = np.round((f_bleu_sum_c_and_diff_c_w + m_bleu_sum_c_and_diff_c_w)/2 - abs(f_bleu_sum_c_and_diff_c_w - m_bleu_sum_c_and_diff_c_w), 1)
            results["BLEU"][translation]["tquality_w_gender_performance"][lset] = bleu_tquality_w_gender_performance

            ## II2. Accuracy
            f_acc_c = results["accuracy"][translation]["feminine"]["correct_ref"]["total"][lset]
            m_acc_c = results["accuracy"][translation]["masculine"]["correct_ref"]["total"][lset]

            ### (II2c) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_f_of_all_c = np.round((f_acc_c / (f_acc_c + m_acc_c)) * 100, 1)
            results["accuracy"][translation]["acc_f_of_all_c"][lset] = acc_f_of_all_c
            ### (II2d) proportion of masculine correct acc. on sum of feminine and masculine correct acc. 
            acc_m_of_all_c = np.round((m_acc_c / (f_acc_c + m_acc_c)) * 100, 1)
            results["accuracy"][translation]["acc_m_of_all_c"][lset] = acc_m_of_all_c
            ### (II2e) = (II2c) - (II2d)
            diff_acc_f_m_of_all_c = np.round(acc_f_of_all_c - acc_m_of_all_c, 1)
            results["accuracy"][translation]["diff_acc_f_m_of_all_c"][lset] = diff_acc_f_m_of_all_c

            f_acc_sum_c_and_diff_c_w = results["accuracy"][translation]["feminine"]["sum_c_and_diff_c_w"][lset]
            m_acc_sum_c_and_diff_c_w = results["accuracy"][translation]["masculine"]["sum_c_and_diff_c_w"][lset]
            ### (II2f) avg. of f(II2b) and m(II2b) - abs. diff. between f(II2b) and m(II2b)
            acc_tquality_w_gender_performance = np.round((f_acc_sum_c_and_diff_c_w + m_acc_sum_c_and_diff_c_w)/2 - abs(f_acc_sum_c_and_diff_c_w - m_acc_sum_c_and_diff_c_w), 1)
            results["accuracy"][translation]["tquality_w_gender_performance"][lset] = acc_tquality_w_gender_performance

            ## II3. Accuracy (category)
            # -> cat. 1
            f_acc_cat_1_c = results["accuracy"][translation]["feminine"]["correct_ref"]["1"][lset]
            m_acc_cat_1_c = results["accuracy"][translation]["masculine"]["correct_ref"]["1"][lset]

            ### (II3c) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_cat_1_f_of_all_c = np.round((f_acc_cat_1_c / (f_acc_cat_1_c + m_acc_cat_1_c)) * 100, 1)
            results["accuracy"][translation]["cat_1_f_of_all_c"][lset] = acc_cat_1_f_of_all_c
            ### (II3d) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_cat_1_m_of_all_c = np.round((m_acc_cat_1_c / (f_acc_cat_1_c + m_acc_cat_1_c)) * 100, 1)
            results["accuracy"][translation]["cat_1_m_of_all_c"][lset] = acc_cat_1_m_of_all_c
            ### (II3e) = (II2c) - (II2d)
            diff_acc_cat_1_f_m_of_all_c = np.round(acc_cat_1_f_of_all_c - acc_cat_1_m_of_all_c, 1)
            results["accuracy"][translation]["cat_1_diff_f_m_of_all_c"][lset] = diff_acc_cat_1_f_m_of_all_c

            f_acc_cat_1_sum_c_and_diff_c_w = results["accuracy"][translation]["feminine"]["cat_1_sum_c_and_diff_c_w"][lset]
            m_acc_cat_1_sum_c_and_diff_c_w = results["accuracy"][translation]["masculine"]["cat_1_sum_c_and_diff_c_w"][lset]
            ### (II3f) avg. of f(II3b) and m(II3b) - abs. diff. between f(II3b) and m(II3b)
            acc_cat_1_tquality_w_gender_performance = np.round((f_acc_cat_1_sum_c_and_diff_c_w + m_acc_cat_1_sum_c_and_diff_c_w)/2 - abs(f_acc_cat_1_sum_c_and_diff_c_w - m_acc_cat_1_sum_c_and_diff_c_w), 1)
            results["accuracy"][translation]["cat_1_tquality_w_gender_performance"][lset] = acc_cat_1_tquality_w_gender_performance

            # -> cat. 2
            f_acc_cat_2_c = results["accuracy"][translation]["feminine"]["correct_ref"]["2"][lset]
            m_acc_cat_2_c = results["accuracy"][translation]["masculine"]["correct_ref"]["2"][lset]

            ### (II3c) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_cat_2_f_of_all_c = np.round((f_acc_cat_2_c / (f_acc_cat_2_c + m_acc_cat_2_c)) * 100, 1)
            results["accuracy"][translation]["cat_2_f_of_all_c"][lset] = acc_cat_2_f_of_all_c
            ### (II3d) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_cat_2_m_of_all_c = np.round((m_acc_cat_2_c / (f_acc_cat_2_c + m_acc_cat_2_c)) * 100, 1)
            results["accuracy"][translation]["cat_2_m_of_all_c"][lset] = acc_cat_2_m_of_all_c
            ### (II3e) = (II2c) - (II2d)
            diff_acc_cat_2_f_m_of_all_c = np.round(acc_cat_2_f_of_all_c - acc_cat_2_m_of_all_c, 1)
            results["accuracy"][translation]["cat_2_diff_f_m_of_all_c"][lset] = diff_acc_cat_2_f_m_of_all_c

            f_acc_cat_2_sum_c_and_diff_c_w = results["accuracy"][translation]["feminine"]["cat_2_sum_c_and_diff_c_w"][lset]
            m_acc_cat_2_sum_c_and_diff_c_w = results["accuracy"][translation]["masculine"]["cat_2_sum_c_and_diff_c_w"][lset]
            ### (II3f) avg. of f(II3b) and m(II3b) - abs. diff. between f(II3b) and m(II3b)
            acc_cat_2_tquality_w_gender_performance = np.round((f_acc_cat_2_sum_c_and_diff_c_w + m_acc_cat_2_sum_c_and_diff_c_w)/2 - abs(f_acc_cat_2_sum_c_and_diff_c_w - m_acc_cat_2_sum_c_and_diff_c_w), 1)
            results["accuracy"][translation]["cat_2_tquality_w_gender_performance"][lset] = acc_cat_2_tquality_w_gender_performance

            ## II4. Accuracy (speaker)
            # -> female
            f_acc_fspeaker_c = results["accuracy"][translation]["feminine"]["correct_ref"]["female_speaker"][lset]
            m_acc_fspeaker_c = results["accuracy"][translation]["masculine"]["correct_ref"]["female_speaker"][lset]

            ### (II4c) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_fspeaker_f_of_all_c = np.round((f_acc_fspeaker_c / (f_acc_fspeaker_c + m_acc_fspeaker_c)) * 100, 1)
            results["accuracy"][translation]["fspeaker_f_of_all_c"][lset] = acc_fspeaker_f_of_all_c
            ### (II4d) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_fspeaker_m_of_all_c = np.round((m_acc_fspeaker_c / (f_acc_fspeaker_c + m_acc_fspeaker_c)) * 100, 1)
            results["accuracy"][translation]["fspeaker_m_of_all_c"][lset] = acc_fspeaker_m_of_all_c
            ### (II4e) = (II2c) - (II2d)
            diff_acc_fspeaker_f_m_of_all_c = np.round(acc_fspeaker_f_of_all_c - acc_fspeaker_m_of_all_c, 1)
            results["accuracy"][translation]["fspeaker_diff_f_m_of_all_c"][lset] = diff_acc_fspeaker_f_m_of_all_c

            f_acc_fspeaker_sum_c_and_diff_c_w = results["accuracy"][translation]["feminine"]["fspeaker_sum_c_and_diff_c_w"][lset]
            m_acc_fspeaker_sum_c_and_diff_c_w = results["accuracy"][translation]["masculine"]["fspeaker_sum_c_and_diff_c_w"][lset]
            ### (II4f) avg. of f(II4b) and m(II4b) - abs. diff. between f(II4b) and m(II4b)
            acc_fspeaker_tquality_w_gender_performance = np.round((f_acc_fspeaker_sum_c_and_diff_c_w + m_acc_fspeaker_sum_c_and_diff_c_w)/2 - abs(f_acc_fspeaker_sum_c_and_diff_c_w - m_acc_fspeaker_sum_c_and_diff_c_w), 1)
            results["accuracy"][translation]["fspeaker_tquality_w_gender_performance"][lset] = acc_fspeaker_tquality_w_gender_performance

            # -> male
            f_acc_mspeaker_c = results["accuracy"][translation]["feminine"]["correct_ref"]["male_speaker"][lset]
            m_acc_mspeaker_c = results["accuracy"][translation]["masculine"]["correct_ref"]["male_speaker"][lset]

            ### (II4c) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_mspeaker_f_of_all_c = np.round((f_acc_mspeaker_c / (f_acc_mspeaker_c + m_acc_mspeaker_c)) * 100, 1)
            results["accuracy"][translation]["mspeaker_f_of_all_c"][lset] = acc_mspeaker_f_of_all_c
            ### (II4d) proportion of feminine correct acc. on sum of feminine and masculine correct acc. 
            acc_mspeaker_m_of_all_c = np.round((m_acc_mspeaker_c / (f_acc_mspeaker_c + m_acc_mspeaker_c)) * 100, 1)
            results["accuracy"][translation]["mspeaker_m_of_all_c"][lset] = acc_mspeaker_m_of_all_c
            ### (II4e) = (II2c) - (II2d)
            diff_acc_mspeaker_f_m_of_all_c = np.round(acc_mspeaker_f_of_all_c - acc_mspeaker_m_of_all_c, 1)
            results["accuracy"][translation]["mspeaker_diff_f_m_of_all_c"][lset] = diff_acc_mspeaker_f_m_of_all_c

            f_acc_mspeaker_sum_c_and_diff_c_w = results["accuracy"][translation]["feminine"]["mspeaker_sum_c_and_diff_c_w"][lset]
            m_acc_mspeaker_sum_c_and_diff_c_w = results["accuracy"][translation]["masculine"]["mspeaker_sum_c_and_diff_c_w"][lset]
            ### (II4f) avg. of f(II4b) and m(II4b) - abs. diff. between f(II4b) and m(II4b)
            acc_mspeaker_tquality_w_gender_performance = np.round((f_acc_mspeaker_sum_c_and_diff_c_w + m_acc_mspeaker_sum_c_and_diff_c_w)/2 - abs(f_acc_mspeaker_sum_c_and_diff_c_w - m_acc_mspeaker_sum_c_and_diff_c_w), 1)
            results["accuracy"][translation]["mspeaker_tquality_w_gender_performance"][lset] = acc_mspeaker_tquality_w_gender_performance

    export_results(results, out_path, train_set)


if __name__ == "__main__":
    main_mustshe()
