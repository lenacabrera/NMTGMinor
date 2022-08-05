import numpy as np
import json
import sys
import os
import argparse
import re
import pandas as pd
import pickle
import os

parser = argparse.ArgumentParser(description='prep_results.py')

parser.add_argument('-raw_path', required=True, default=None)
parser.add_argument('-pred_path', required=True, default=None)
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

    return accuracies_total, accuracies_1, accuracies_2, accuracies_f_speaker, accuracies_m_speaker

def get_avg_accuracies(accuracies_total, accuracies_1, accuracies_2, accuracies_f_speaker, accuracies_m_speaker):
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

def get_empty_results_dict():
    results = {
        "BLEU": {
            "zero_shot": {
                "all": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "feminine": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "masculine": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "f_of_all_c": {},
                "m_of_all_c": {},
                "diff_f_m_of_all_c": {},
                "tquality_w_gender_performance": {},
            },
            "pivot": {
                "all": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "feminine": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "masculine": {
                    "correct_ref": {},
                    "wrong_ref": {},
                    "diff_c_w": {},
                    "sum_c_and_diff_c_w": {},
                },
                "f_of_all_c": {},
                "m_of_all_c": {},
                "diff_f_m_of_all_c": {},
                "tquality_w_gender_performance": {},
            }
        },
        "accuracy": {
            "zero_shot": {
                "total": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
                "1": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
                "2": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
                "female_speaker": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
                "male_speaker": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
            },
            "pivot": {
                "total": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
                "1": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
                "2": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
                "female_speaker": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
                "male_speaker": {
                    "all": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "feminine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "masculine": {
                        "correct_ref": {},
                        "wrong_ref": {},
                        "diff_c_w": {},
                        "sum_c_and_diff_c_w": {}
                    },
                    "f_of_all_c": {},
                    "m_of_all_c": {},
                    "diff_f_m_of_all_c": {},
                    "tquality_w_gender_performance": {},
                },
            }
        }
    }

    return results

def calc_and_store_results_per_lset(results, raw_path, pred_path):
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
                                acc_total_zs, acc_1_zs, acc_2_zs, acc_f_zs, acc_m_zs = get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl=None)
                                avg_acc_total_zs, avg_acc_1_zs, avg_acc_2_zs, avg_acc_f_zs, avg_acc_m_zs = get_avg_accuracies(acc_total_zs, acc_1_zs, acc_2_zs, acc_f_zs, acc_m_zs)
                                results["accuracy"][translation]["total"][gender_set][ref][lset] = avg_acc_total_zs
                                results["accuracy"][translation]["1"][gender_set][ref][lset] = avg_acc_1_zs
                                results["accuracy"][translation]["2"][gender_set][ref][lset] = avg_acc_2_zs
                                results["accuracy"][translation]["female_speaker"][gender_set][ref][lset] = avg_acc_f_zs
                                results["accuracy"][translation]["male_speaker"][gender_set][ref][lset] = avg_acc_m_zs
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
                                    acc_total_pv, acc_1_pv, acc_2_pv, acc_f_pv, acc_m_pv = get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl)
                                    avg_acc_total_pv, avg_acc_1_pv, avg_acc_2_pv, avg_acc_f_pv, avg_acc_m_pv = get_avg_accuracies(acc_total_pv, acc_1_pv, acc_2_pv, acc_f_pv, acc_m_pv)
                                    results["accuracy"][translation]["total"][gender_set][ref][lset] = avg_acc_total_pv
                                    results["accuracy"][translation]["1"][gender_set][ref][lset] = avg_acc_1_pv
                                    results["accuracy"][translation]["2"][gender_set][ref][lset] = avg_acc_2_pv
                                    results["accuracy"][translation]["female_speaker"][gender_set][ref][lset] = avg_acc_f_pv
                                    results["accuracy"][translation]["male_speaker"][gender_set][ref][lset] = avg_acc_m_pv
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
                results = calc_1__diff_c_w(results, "BLEU", translation, gender_set, lset)
                results = calc_2__sum_c_and_diff_c_w(results, "BLEU", translation, gender_set, lset)

                ## I2. Accuracy (total)
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="total")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="total")
                              
                ## I3. Accuracy (category)
                # -> cat. 1
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="1")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="1")
                # -> cat. 2
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="2")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="2")

                ## I4. Accuracy (speaker)
                # -> female
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="female_speaker")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="female_speaker")
                # -> male
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="male_speaker")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, lset, acc_type="male_speaker")

        # additional metrics (II)
        for lset in lsets:
            ## II2. BLEU
            results = calc_3__f_m_of_all_c(results, "BLEU", translation, lset)
            results = calc_4__diff_f_m_of_all_c(results, "BLEU", translation, lset)
            results = calc_5__tradeoff_metric_diff(results, "BLEU", translation, lset)

            ## II2. Accuracy (total)
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, lset, acc_type="total")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, lset, acc_type="total")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, lset, acc_type="total")

            ## II3. Accuracy (category)
            # -> cat. 1
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, lset, acc_type="1")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, lset, acc_type="1")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, lset, acc_type="1")
            # -> cat. 2
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, lset, acc_type="2")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, lset, acc_type="2")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, lset, acc_type="2")

            ## II4. Accuracy (speaker)
            # -> female
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, lset, acc_type="female_speaker")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, lset, acc_type="female_speaker")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, lset, acc_type="female_speaker")
            # -> male
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, lset, acc_type="male_speaker")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, lset, acc_type="male_speaker")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, lset, acc_type="male_speaker")

    return results

def calc_and_store_results_avg_zeroshot_directions(results, raw_path, pred_path):

    for translation in ["zero_shot", "pivot"]:
        for gender_set in ["all", "feminine", "masculine"]:
            for ref in ["correct_ref", "wrong_ref"]:
                if translation == "zero_shot":
                    # zero-shot
                    for f in os.listdir(f"{pred_path}/{ref}/{gender_set}"):
                        if os.path.isfile(os.path.join(f"{pred_path}/{ref}/{gender_set}", f)):
                            lset = re.search(r"[a-z][a-z]-[a-z][a-z]", os.path.basename(f)).group(0)
                            sl = lset.split("-")[0]
                            tl = lset.split("-")[1]
                            if sl in ["fr", "it"] and tl in ["fr", "it"]:
                                # zero-shot direction
                                if f.endswith(".res"):
                                    # BLEU
                                    lines_zs = open(f"{pred_path}/{ref}/{gender_set}/{f}").readlines() 
                                    bleu_zs = get_bleu_scores_mustshe(lines_zs)
                                    if "zs_avg" not in results["BLEU"][translation][gender_set][ref]:
                                        results["BLEU"][translation][gender_set][ref]["zs_avg"] = []
                                    results["BLEU"][translation][gender_set][ref]["zs_avg"].append(float(bleu_zs))
                                elif f.startswith(lset) and f.endswith(".pt"):
                                    # Accuracy
                                    if "zs_avg" not in results["accuracy"][translation]["total"][gender_set][ref]:
                                        results["accuracy"][translation]["total"][gender_set][ref]["zs_avg"] = []
                                        results["accuracy"][translation]["1"][gender_set][ref]["zs_avg"] = []
                                        results["accuracy"][translation]["2"][gender_set][ref]["zs_avg"] = []
                                        results["accuracy"][translation]["female_speaker"][gender_set][ref]["zs_avg"] = []
                                        results["accuracy"][translation]["male_speaker"][gender_set][ref]["zs_avg"] = []
                                    acc_total_zs, acc_1_zs, acc_2_zs, acc_f_zs, acc_m_zs = get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl=None)
                                    results["accuracy"][translation]["total"][gender_set][ref]["zs_avg"].append(acc_total_zs)
                                    results["accuracy"][translation]["1"][gender_set][ref]["zs_avg"].append(acc_1_zs)
                                    results["accuracy"][translation]["2"][gender_set][ref]["zs_avg"].append(acc_2_zs)
                                    results["accuracy"][translation]["female_speaker"][gender_set][ref]["zs_avg"].append(acc_f_zs)
                                    results["accuracy"][translation]["male_speaker"][gender_set][ref]["zs_avg"].append(acc_m_zs)
                                else:
                                    continue
                            else:
                                # not zero-shot direction
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
                            if sl != pl and tl != pl:
                                if sl in ["fr", "it"] and tl in ["fr", "it"]:
                                    # zero-shot direction
                                    if f.endswith(".res"):
                                        # BLEU
                                        lines_pv = open(f"{pred_path}/pivot/{ref}/{gender_set}/{sl}-{pl}-{tl}.real.pivotout.t.res").readlines()
                                        bleu_pv = get_bleu_scores_mustshe(lines_pv)
                                        if "zs_avg" not in results["BLEU"][translation][gender_set][ref]:
                                            results["BLEU"][translation][gender_set][ref]["zs_avg"] = []
                                        results["BLEU"][translation][gender_set][ref]["zs_avg"].append(float(bleu_pv))
                                    elif f.startswith(f"{sl}-{pl}-{tl}") and f.endswith(".pt"):
                                    # Accuracy
                                        if "zs_avg" not in results["accuracy"][translation]["total"][gender_set][ref]:
                                            results["accuracy"][translation]["total"][gender_set][ref]["zs_avg"] = []
                                            results["accuracy"][translation]["1"][gender_set][ref]["zs_avg"] = []
                                            results["accuracy"][translation]["2"][gender_set][ref]["zs_avg"] = []
                                            results["accuracy"][translation]["female_speaker"][gender_set][ref]["zs_avg"] = []
                                            results["accuracy"][translation]["male_speaker"][gender_set][ref]["zs_avg"] = []
                                        acc_total_pv, acc_1_pv, acc_2_pv, acc_f_pv, acc_m_pv = get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl)
                                        results["accuracy"][translation]["total"][gender_set][ref]["zs_avg"].append(acc_total_pv)
                                        results["accuracy"][translation]["1"][gender_set][ref]["zs_avg"].append(acc_1_pv)
                                        results["accuracy"][translation]["2"][gender_set][ref]["zs_avg"].append(acc_2_pv)
                                        results["accuracy"][translation]["female_speaker"][gender_set][ref]["zs_avg"].append(acc_f_pv)
                                        results["accuracy"][translation]["male_speaker"][gender_set][ref]["zs_avg"].append(acc_m_pv)
                                    else:
                                        continue   
                                else:
                                    # not zero-shot direction
                                    continue
                            else:
                                continue

    for translation in ["zero_shot", "pivot"]:
        for gender_set in ["all", "feminine", "masculine"]:
            for ref in ["correct_ref", "wrong_ref"]:
                bleu_avg = np.round(np.average(results["BLEU"][translation][gender_set][ref]["zs_avg"]), 1)
                acc_total_avg = np.round(np.average(results["accuracy"][translation]["total"][gender_set][ref]["zs_avg"]) * 100, 1)
                acc_1_avg = np.round(np.average(results["accuracy"][translation]["1"][gender_set][ref]["zs_avg"]) * 100, 1)
                acc_2_avg = np.round(np.average(results["accuracy"][translation]["2"][gender_set][ref]["zs_avg"]) * 100, 1)
                acc_fspeaker_avg = np.round(np.average(results["accuracy"][translation]["female_speaker"][gender_set][ref]["zs_avg"]) * 100, 1)
                acc_mspeaker_avg = np.round(np.average(results["accuracy"][translation]["male_speaker"][gender_set][ref]["zs_avg"]) * 100, 1)

                results["BLEU"][translation][gender_set][ref]["zs_avg"] = bleu_avg
                results["accuracy"][translation]["total"][gender_set][ref]["zs_avg"] = acc_total_avg
                results["accuracy"][translation]["1"][gender_set][ref]["zs_avg"] = acc_1_avg
                results["accuracy"][translation]["2"][gender_set][ref]["zs_avg"] = acc_2_avg
                results["accuracy"][translation]["female_speaker"][gender_set][ref]["zs_avg"] = acc_fspeaker_avg
                results["accuracy"][translation]["male_speaker"][gender_set][ref]["zs_avg"] = acc_mspeaker_avg

            # additional metrics (I)
            ## I1. BLEU
            results = calc_1__diff_c_w(results, "BLEU", translation, gender_set, "zs_avg")
            results = calc_2__sum_c_and_diff_c_w(results, "BLEU", translation, gender_set, "zs_avg")

            ## I2. Accuracy (total)
            results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="total")
            results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="total")
                        
            ## I3. Accuracy (category)
            # -> cat. 1
            results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="1")
            results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="1")
            # -> cat. 2
            results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="2")
            results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="2")

            ## I4. Accuracy (speaker)
            # -> female
            results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="female_speaker")
            results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="female_speaker")
            # -> male
            results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="male_speaker")
            results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "zs_avg", acc_type="male_speaker")


        # additional metrics (II)
        ## II2. BLEU
        results = calc_3__f_m_of_all_c(results, "BLEU", translation, "zs_avg")
        results = calc_4__diff_f_m_of_all_c(results, "BLEU", translation, "zs_avg")
        results = calc_5__tradeoff_metric_diff(results, "BLEU", translation, "zs_avg")

        ## II2. Accuracy (total)
        results = calc_3__f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="total")
        results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="total")
        results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "zs_avg", acc_type="total")

        ## II3. Accuracy (category)
        # -> cat. 1
        results = calc_3__f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="1")
        results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="1")
        results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "zs_avg", acc_type="1")
        # -> cat. 2
        results = calc_3__f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="2")
        results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="2")
        results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "zs_avg", acc_type="2")

        ## II4. Accuracy (speaker)
        # -> female
        results = calc_3__f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="female_speaker")
        results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="female_speaker")
        results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "zs_avg", acc_type="female_speaker")
        # -> male
        results = calc_3__f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="male_speaker")
        results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "zs_avg", acc_type="male_speaker")
        results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "zs_avg", acc_type="male_speaker")

    return results

def calc_and_store_results_avg_supervised_directions(results, raw_path, pred_path, train_set):

    train_lang = []
    if "EN" in train_set or train_set == "twoway.r32.q" or train_set == "twoway.r32.q.new":
        train_lang.append("en")
    elif "ES" in train_set:
        train_lang.append("es")
    elif "DE" in train_set:
        train_lang.append("de")
    elif "FR" in train_set:
        train_lang.append("fr")
    elif "IT" in train_set:
        train_lang.append("it")

    for translation in ["zero_shot", "pivot"]:
        for gender_set in ["all", "feminine", "masculine"]:
            for ref in ["correct_ref", "wrong_ref"]:
                if translation == "zero_shot":
                    # zero-shot
                    for f in os.listdir(f"{pred_path}/{ref}/{gender_set}"):
                        if os.path.isfile(os.path.join(f"{pred_path}/{ref}/{gender_set}", f)):
                            lset = re.search(r"[a-z][a-z]-[a-z][a-z]", os.path.basename(f)).group(0)
                            sl = lset.split("-")[0]
                            tl = lset.split("-")[1]
                            
                            if sl in train_lang or tl in train_lang:
                                # zero-shot direction
                                if f.endswith(".res"):
                                    # BLEU
                                    lines_zs = open(f"{pred_path}/{ref}/{gender_set}/{f}").readlines() 
                                    bleu_zs = get_bleu_scores_mustshe(lines_zs)
                                    if "sv_avg" not in results["BLEU"][translation][gender_set][ref]:
                                        results["BLEU"][translation][gender_set][ref]["sv_avg"] = []
                                    results["BLEU"][translation][gender_set][ref]["sv_avg"].append(float(bleu_zs))
                                elif f.startswith(lset) and f.endswith(".pt"):
                                    # Accuracy
                                    if "sv_avg" not in results["accuracy"][translation]["total"][gender_set][ref]:
                                        results["accuracy"][translation]["total"][gender_set][ref]["sv_avg"] = []
                                        results["accuracy"][translation]["1"][gender_set][ref]["sv_avg"] = []
                                        results["accuracy"][translation]["2"][gender_set][ref]["sv_avg"] = []
                                        results["accuracy"][translation]["female_speaker"][gender_set][ref]["sv_avg"] = []
                                        results["accuracy"][translation]["male_speaker"][gender_set][ref]["sv_avg"] = []
                                    acc_total_zs, acc_1_zs, acc_2_zs, acc_f_zs, acc_m_zs = get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl=None)
                                    results["accuracy"][translation]["total"][gender_set][ref]["sv_avg"].append(acc_total_zs)
                                    results["accuracy"][translation]["1"][gender_set][ref]["sv_avg"].append(acc_1_zs)
                                    results["accuracy"][translation]["2"][gender_set][ref]["sv_avg"].append(acc_2_zs)
                                    results["accuracy"][translation]["female_speaker"][gender_set][ref]["sv_avg"].append(acc_f_zs)
                                    results["accuracy"][translation]["male_speaker"][gender_set][ref]["sv_avg"].append(acc_m_zs)
                                else:
                                    continue
                            else:
                                # not zero-shot direction
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
                            if sl != pl and tl != pl:
                                if sl in train_lang or tl in train_lang:
                                    # zero-shot direction
                                    if f.endswith(".res"):
                                        # BLEU
                                        lines_pv = open(f"{pred_path}/pivot/{ref}/{gender_set}/{sl}-{pl}-{tl}.real.pivotout.t.res").readlines()
                                        bleu_pv = get_bleu_scores_mustshe(lines_pv)
                                        if "sv_avg" not in results["BLEU"][translation][gender_set][ref]:
                                            results["BLEU"][translation][gender_set][ref]["sv_avg"] = []
                                        results["BLEU"][translation][gender_set][ref]["sv_avg"].append(float(bleu_pv))
                                    elif f.startswith(f"{sl}-{pl}-{tl}") and f.endswith(".pt"):
                                    # Accuracy
                                        if "sv_avg" not in results["accuracy"][translation]["total"][gender_set][ref]:
                                            results["accuracy"][translation]["total"][gender_set][ref]["sv_avg"] = []
                                            results["accuracy"][translation]["1"][gender_set][ref]["sv_avg"] = []
                                            results["accuracy"][translation]["2"][gender_set][ref]["sv_avg"] = []
                                            results["accuracy"][translation]["female_speaker"][gender_set][ref]["sv_avg"] = []
                                            results["accuracy"][translation]["male_speaker"][gender_set][ref]["sv_avg"] = []
                                        acc_total_pv, acc_1_pv, acc_2_pv, acc_f_pv, acc_m_pv = get_accuracies_mustshe(raw_path, pred_path, ref, gender_set, f, sl, tl, pl)
                                        results["accuracy"][translation]["total"][gender_set][ref]["sv_avg"].append(acc_total_pv)
                                        results["accuracy"][translation]["1"][gender_set][ref]["sv_avg"].append(acc_1_pv)
                                        results["accuracy"][translation]["2"][gender_set][ref]["sv_avg"].append(acc_2_pv)
                                        results["accuracy"][translation]["female_speaker"][gender_set][ref]["sv_avg"].append(acc_f_pv)
                                        results["accuracy"][translation]["male_speaker"][gender_set][ref]["sv_avg"].append(acc_m_pv)
                                    else:
                                        continue   
                                else:
                                    # not zero-shot direction
                                    continue
                            else:
                                continue

    for translation in ["zero_shot", "pivot"]:
        for gender_set in ["all", "feminine", "masculine"]:
            for ref in ["correct_ref", "wrong_ref"]:
                if "sv_avg" in results["BLEU"][translation][gender_set][ref]:
                    bleu_avg = np.round(np.average(results["BLEU"][translation][gender_set][ref]["sv_avg"]), 1)
                    results["BLEU"][translation][gender_set][ref]["sv_avg"] = bleu_avg
                if "sv_avg" in results["accuracy"][translation]["total"][gender_set][ref]:
                    acc_total_avg = np.round(np.average(results["accuracy"][translation]["total"][gender_set][ref]["sv_avg"]) * 100, 1)
                    acc_1_avg = np.round(np.average(results["accuracy"][translation]["1"][gender_set][ref]["sv_avg"]) * 100, 1)
                    acc_2_avg = np.round(np.average(results["accuracy"][translation]["2"][gender_set][ref]["sv_avg"]) * 100, 1)
                    acc_fspeaker_avg = np.round(np.average(results["accuracy"][translation]["female_speaker"][gender_set][ref]["sv_avg"]) * 100, 1)
                    acc_mspeaker_avg = np.round(np.average(results["accuracy"][translation]["male_speaker"][gender_set][ref]["sv_avg"]) * 100, 1)

                    results["accuracy"][translation]["total"][gender_set][ref]["sv_avg"] = acc_total_avg
                    results["accuracy"][translation]["1"][gender_set][ref]["sv_avg"] = acc_1_avg
                    results["accuracy"][translation]["2"][gender_set][ref]["sv_avg"] = acc_2_avg
                    results["accuracy"][translation]["female_speaker"][gender_set][ref]["sv_avg"] = acc_fspeaker_avg
                    results["accuracy"][translation]["male_speaker"][gender_set][ref]["sv_avg"] = acc_mspeaker_avg

            # additional metrics (I)
            ## I1. BLEU
            if "sv_avg" in results["BLEU"][translation][gender_set]["correct_ref"]:
                results = calc_1__diff_c_w(results, "BLEU", translation, gender_set, "sv_avg")
                results = calc_2__sum_c_and_diff_c_w(results, "BLEU", translation, gender_set, "sv_avg")

            ## I2. Accuracy (total)
            if "sv_avg" in results["accuracy"][translation]["total"][gender_set]["correct_ref"]:
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="total")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="total")
                        
            ## I3. Accuracy (category)
            # -> cat. 1
            if "sv_avg" in results["accuracy"][translation]["1"][gender_set]["correct_ref"]:
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="1")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="1")
            # -> cat. 2
            if "sv_avg" in results["accuracy"][translation]["2"][gender_set]["correct_ref"]:
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="2")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="2")

            ## I4. Accuracy (speaker)
            # -> female
            if "sv_avg" in results["accuracy"][translation]["female_speaker"][gender_set]["correct_ref"]:
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="female_speaker")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="female_speaker")
            # -> male
            if "sv_avg" in results["accuracy"][translation]["male_speaker"][gender_set]["correct_ref"]:
                results = calc_1__diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="male_speaker")
                results = calc_2__sum_c_and_diff_c_w(results, "accuracy", translation, gender_set, "sv_avg", acc_type="male_speaker")


        # additional metrics (II)
        ## II2. BLEU
        if "sv_avg" in results["BLEU"][translation]["all"]["correct_ref"]:
            results = calc_3__f_m_of_all_c(results, "BLEU", translation, "sv_avg")
            results = calc_4__diff_f_m_of_all_c(results, "BLEU", translation, "sv_avg")
            results = calc_5__tradeoff_metric_diff(results, "BLEU", translation, "sv_avg")

        ## II2. Accuracy (total)
        if "sv_avg" in results["accuracy"][translation]["total"]["all"]["correct_ref"]:
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="total")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="total")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "sv_avg", acc_type="total")

        ## II3. Accuracy (category)
        # -> cat. 1
        if "sv_avg" in results["accuracy"][translation]["1"]["all"]["correct_ref"]:
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="1")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="1")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "sv_avg", acc_type="1")
        # -> cat. 2
        if "sv_avg" in results["accuracy"][translation]["2"]["all"]["correct_ref"]:
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="2")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="2")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "sv_avg", acc_type="2")

        ## II4. Accuracy (speaker)
        # -> female
        if "sv_avg" in results["accuracy"][translation]["female_speaker"]["all"]["correct_ref"]:
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="female_speaker")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="female_speaker")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "sv_avg", acc_type="female_speaker")
        # -> male
        if "sv_avg" in results["accuracy"][translation]["male_speaker"]["all"]["correct_ref"]:
            results = calc_3__f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="male_speaker")
            results = calc_4__diff_f_m_of_all_c(results, "accuracy", translation, "sv_avg", acc_type="male_speaker")
            results = calc_5__tradeoff_metric_diff(results, "accuracy", translation, "sv_avg", acc_type="male_speaker")

    return results

def calc_1__diff_c_w(results, metric, translation, gender_set, lset, acc_type=None):
    if metric == "accuracy":
        results[metric][translation][acc_type][gender_set]["diff_c_w"][lset] = np.round(results[metric][translation][acc_type][gender_set]["correct_ref"][lset] - \
            results[metric][translation][acc_type][gender_set]["wrong_ref"][lset], 1)
    else:
        results[metric][translation][gender_set]["diff_c_w"][lset] = np.round(results[metric][translation][gender_set]["correct_ref"][lset] - \
            results[metric][translation][gender_set]["wrong_ref"][lset], 1)
    return results

def calc_2__sum_c_and_diff_c_w(results, metric, translation, gender_set, lset, acc_type=None):
    if metric == "accuracy":
        results[metric][translation][acc_type][gender_set]["sum_c_and_diff_c_w"][lset] =  np.round(results[metric][translation][acc_type][gender_set]["correct_ref"][lset] + \
            results[metric][translation][acc_type][gender_set]["diff_c_w"][lset], 1)
    else:
        results[metric][translation][gender_set]["sum_c_and_diff_c_w"][lset] = np.round(results[metric][translation][gender_set]["correct_ref"][lset] + \
            results[metric][translation][gender_set]["diff_c_w"][lset], 1)
    return results

def calc_3__f_m_of_all_c(results, metric, translation, lset, acc_type=None):
    if metric == "accuracy":
        if lset != "sv_avg" or "sv_avg" in results[metric][translation][acc_type]["feminine"]["correct_ref"]:
            f_c = results[metric][translation][acc_type]["feminine"]["correct_ref"][lset]
            m_c = results[metric][translation][acc_type]["masculine"]["correct_ref"][lset]
            results[metric][translation][acc_type]["f_of_all_c"][lset] = np.round((f_c / (f_c + m_c)) * 100, 1)
            results[metric][translation][acc_type]["m_of_all_c"][lset] = np.round((m_c / (f_c + m_c)) * 100, 1)
    else:
        if lset != "sv_avg" or "sv_avg" in results[metric][translation]["feminine"]["correct_ref"]:
            f_c = results[metric][translation]["feminine"]["correct_ref"][lset]
            m_c = results[metric][translation]["masculine"]["correct_ref"][lset]
            results[metric][translation]["f_of_all_c"][lset] = np.round((f_c / (f_c + m_c)) * 100, 1)
            results[metric][translation]["m_of_all_c"][lset] = np.round((m_c / (f_c + m_c)) * 100, 1)
    return results

def calc_4__diff_f_m_of_all_c(results, metric, translation, lset, acc_type=None):
    if metric == "accuracy":
        if lset != "sv_avg" or "sv_avg" in results[metric][translation][acc_type]["f_of_all_c"]:
            f_of_all_c = results[metric][translation][acc_type]["f_of_all_c"][lset]
            m_of_all_c = results[metric][translation][acc_type]["m_of_all_c"][lset]
            results[metric][translation][acc_type]["diff_f_m_of_all_c"][lset] = np.round(f_of_all_c - m_of_all_c, 1)
    else:
        if lset != "sv_avg" or "sv_avg" in results[metric][translation]["f_of_all_c"]:
            f_of_all_c = results[metric][translation]["f_of_all_c"][lset]
            m_of_all_c = results[metric][translation]["m_of_all_c"][lset]
            results[metric][translation]["diff_f_m_of_all_c"][lset] = np.round(f_of_all_c - m_of_all_c, 1)
    return results

def calc_5__tradeoff_metric_diff(results, metric, translation, lset, acc_type=None):
    if metric == "accuracy":
        if lset != "sv_avg" or "sv_avg" in results[metric][translation][acc_type]["feminine"]["sum_c_and_diff_c_w"]:
            f_sum_c_and_diff_c_w = results[metric][translation][acc_type]["feminine"]["sum_c_and_diff_c_w"][lset]
            m_sum_c_and_diff_c_w = results[metric][translation][acc_type]["masculine"]["sum_c_and_diff_c_w"][lset]
            results[metric][translation][acc_type]["tquality_w_gender_performance"][lset] = np.round((f_sum_c_and_diff_c_w + m_sum_c_and_diff_c_w)/2 - \
                abs(f_sum_c_and_diff_c_w - m_sum_c_and_diff_c_w), 1)
    else:
        if lset != "sv_avg" or "sv_avg" in results[metric][translation]["feminine"]["sum_c_and_diff_c_w"]:
            f_sum_c_and_diff_c_w = results[metric][translation]["feminine"]["sum_c_and_diff_c_w"][lset]
            m_sum_c_and_diff_c_w = results[metric][translation]["masculine"]["sum_c_and_diff_c_w"][lset]
            results[metric][translation]["tquality_w_gender_performance"][lset] = np.round((f_sum_c_and_diff_c_w + m_sum_c_and_diff_c_w)/2 - \
                abs(f_sum_c_and_diff_c_w - m_sum_c_and_diff_c_w), 1)
    return results
    
def export_results(results, metric, df, out_path, map_train_set_model_name, train_set, acc_type=None, avg_sv=False):
    sl = ''
    tl = ''
    cur_model = ""
    if metric == "BLEU" or acc_type == "total":
        num_rows = 12 * len(map_train_set_model_name) + 2
    else:
        num_rows = 2 * 12 * len(map_train_set_model_name) + 2

    i = -1
    for _ , r in df.iterrows():
        i += 1
        if i < num_rows:
            if i < 2:
                # skip header rows
                continue
            else:
                if sl == '' or not pd.isna(df.loc[i, "sl"]):
                    sl = df.loc[i, "sl"]
                if tl == '' or not pd.isna(df.loc[i, "tl"]):
                    tl = r["tl"]
                if cur_model == "" or not pd.isna(df.loc[i, "model"]):
                    cur_model = r["model"]

                if map_train_set_model_name[train_set] == cur_model:
                    if metric == "BLEU":
                        # check if sl-tl pair is in results
                        if f"{sl}-{tl}" in results[metric]["zero_shot"]["all"]["correct_ref"]:
                            # all
                            df.loc[i, "all_cor_zs"] = results[metric]["zero_shot"]["all"]["correct_ref"][f"{sl}-{tl}"]
                            df.loc[i, "all_wro_zs"] = results[metric]["zero_shot"]["all"]["wrong_ref"][f"{sl}-{tl}"]
                            df.loc[i, "all_diff_zs"] = results[metric]["zero_shot"]["all"]["diff_c_w"][f"{sl}-{tl}"]
                            df.loc[i, "all_sum_diff_zs"] = results[metric]["zero_shot"]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                            # feminine
                            df.loc[i, "f_cor_zs"] = results[metric]["zero_shot"]["feminine"]["correct_ref"][f"{sl}-{tl}"]
                            df.loc[i, "f_wro_zs"] = results[metric]["zero_shot"]["feminine"]["wrong_ref"][f"{sl}-{tl}"]
                            df.loc[i, "f_diff_zs"] = results[metric]["zero_shot"]["feminine"]["diff_c_w"][f"{sl}-{tl}"]
                            df.loc[i, "f_sum_diff_zs"] = results[metric]["zero_shot"]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                            # masculine
                            df.loc[i, "m_cor_zs"] = results[metric]["zero_shot"]["masculine"]["correct_ref"][f"{sl}-{tl}"]
                            df.loc[i, "m_wro_zs"] = results[metric]["zero_shot"]["masculine"]["wrong_ref"][f"{sl}-{tl}"]
                            df.loc[i, "m_diff_zs"] = results[metric]["zero_shot"]["masculine"]["diff_c_w"][f"{sl}-{tl}"]
                            df.loc[i, "m_sum_diff_zs"] = results[metric]["zero_shot"]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                            # additional metrics
                            df.loc[i, "f_of_cor_zs"] = results[metric]["zero_shot"]["f_of_all_c"][f"{sl}-{tl}"]
                            df.loc[i, "m_of_cor_zs"] = results[metric]["zero_shot"]["m_of_all_c"][f"{sl}-{tl}"]
                            df.loc[i, "diff_f_m_of_cor_zs"] = results[metric]["zero_shot"]["diff_f_m_of_all_c"][f"{sl}-{tl}"]
                            df.loc[i, "gatq_zs"] = results[metric]["zero_shot"]["tquality_w_gender_performance"][f"{sl}-{tl}"]
                        if f"{sl}-{tl}" in results[metric]["pivot"]["all"]["correct_ref"]:
                            # all
                            df.loc[i, "all_cor_pv"] = results[metric]["pivot"]["all"]["correct_ref"][f"{sl}-{tl}"]
                            df.loc[i, "all_wro_pv"] = results[metric]["pivot"]["all"]["wrong_ref"][f"{sl}-{tl}"]
                            df.loc[i, "all_diff_pv"] = results[metric]["pivot"]["all"]["diff_c_w"][f"{sl}-{tl}"]
                            df.loc[i, "all_sum_diff_pv"] = results[metric]["pivot"]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                            # feminine
                            df.loc[i, "f_cor_pv"] = results[metric]["pivot"]["feminine"]["correct_ref"][f"{sl}-{tl}"]
                            df.loc[i, "f_wro_pv"] = results[metric]["pivot"]["feminine"]["wrong_ref"][f"{sl}-{tl}"]
                            df.loc[i, "f_diff_pv"] = results[metric]["pivot"]["feminine"]["diff_c_w"][f"{sl}-{tl}"]
                            df.loc[i, "f_sum_diff_pv"] = results[metric]["pivot"]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                            # masculine
                            df.loc[i, "m_cor_pv"] = results[metric]["pivot"]["masculine"]["correct_ref"][f"{sl}-{tl}"]
                            df.loc[i, "m_wro_pv"] = results[metric]["pivot"]["masculine"]["wrong_ref"][f"{sl}-{tl}"]
                            df.loc[i, "m_diff_pv"] = results[metric]["pivot"]["masculine"]["diff_c_w"][f"{sl}-{tl}"]
                            df.loc[i, "m_sum_diff_pv"] = results[metric]["pivot"]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                            # additional metrics
                            df.loc[i, "f_of_cor_pv"] = results[metric]["pivot"]["f_of_all_c"][f"{sl}-{tl}"]
                            df.loc[i, "m_of_cor_pv"] = results[metric]["pivot"]["m_of_all_c"][f"{sl}-{tl}"]
                            df.loc[i, "diff_f_m_of_cor_pv"] = results[metric]["pivot"]["diff_f_m_of_all_c"][f"{sl}-{tl}"]
                            df.loc[i, "gatq_pv"] = results[metric]["pivot"]["tquality_w_gender_performance"][f"{sl}-{tl}"]

                    if metric == "accuracy":
                        # print(sl, tl)
                        if i % 2 != 0 and (acc_type == "1" or acc_type == "female_speaker"):
                            continue
                        else:
                            # check if sl-tl pair is in results
                            if f"{sl}-{tl}" in results[metric]["zero_shot"][acc_type]["all"]["correct_ref"]:
                                # all
                                df.loc[i, "all_cor_zs"] = results[metric]["zero_shot"][acc_type]["all"]["correct_ref"][f"{sl}-{tl}"]
                                df.loc[i, "all_wro_zs"] = results[metric]["zero_shot"][acc_type]["all"]["wrong_ref"][f"{sl}-{tl}"]
                                df.loc[i, "all_diff_zs"] = results[metric]["zero_shot"][acc_type]["all"]["diff_c_w"][f"{sl}-{tl}"]
                                df.loc[i, "all_sum_diff_zs"] = results[metric]["zero_shot"][acc_type]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                # feminine
                                df.loc[i, "f_cor_zs"] = results[metric]["zero_shot"][acc_type]["feminine"]["correct_ref"][f"{sl}-{tl}"]
                                df.loc[i, "f_wro_zs"] = results[metric]["zero_shot"][acc_type]["feminine"]["wrong_ref"][f"{sl}-{tl}"]
                                df.loc[i, "f_diff_zs"] = results[metric]["zero_shot"][acc_type]["feminine"]["diff_c_w"][f"{sl}-{tl}"]
                                df.loc[i, "f_sum_diff_zs"] = results[metric]["zero_shot"][acc_type]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                # masculine
                                df.loc[i, "m_cor_zs"] = results[metric]["zero_shot"][acc_type]["masculine"]["correct_ref"][f"{sl}-{tl}"]
                                df.loc[i, "m_wro_zs"] = results[metric]["zero_shot"][acc_type]["masculine"]["wrong_ref"][f"{sl}-{tl}"]
                                df.loc[i, "m_diff_zs"] = results[metric]["zero_shot"][acc_type]["masculine"]["diff_c_w"][f"{sl}-{tl}"]
                                df.loc[i, "m_sum_diff_zs"] = results[metric]["zero_shot"][acc_type]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                # additional metrics
                                df.loc[i, "f_of_cor_zs"] = results[metric]["zero_shot"][acc_type]["f_of_all_c"][f"{sl}-{tl}"]
                                df.loc[i, "m_of_cor_zs"] = results[metric]["zero_shot"][acc_type]["m_of_all_c"][f"{sl}-{tl}"]
                                df.loc[i, "diff_f_m_of_cor_zs"] = results[metric]["zero_shot"][acc_type]["diff_f_m_of_all_c"][f"{sl}-{tl}"]
                                df.loc[i, "gatq_zs"] = results[metric]["zero_shot"][acc_type]["tquality_w_gender_performance"][f"{sl}-{tl}"]
                            if f"{sl}-{tl}" in results[metric]["pivot"][acc_type]["all"]["correct_ref"]:
                                # all
                                df.loc[i, "all_cor_pv"] = results[metric]["pivot"][acc_type]["all"]["correct_ref"][f"{sl}-{tl}"]
                                df.loc[i, "all_wro_pv"] = results[metric]["pivot"][acc_type]["all"]["wrong_ref"][f"{sl}-{tl}"]
                                df.loc[i, "all_diff_pv"] = results[metric]["pivot"][acc_type]["all"]["diff_c_w"][f"{sl}-{tl}"]
                                df.loc[i, "all_sum_diff_pv"] = results[metric]["pivot"][acc_type]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                # feminine
                                df.loc[i, "f_cor_pv"] = results[metric]["pivot"][acc_type]["feminine"]["correct_ref"][f"{sl}-{tl}"]
                                df.loc[i, "f_wro_pv"] = results[metric]["pivot"][acc_type]["feminine"]["wrong_ref"][f"{sl}-{tl}"]
                                df.loc[i, "f_diff_pv"] = results[metric]["pivot"][acc_type]["feminine"]["diff_c_w"][f"{sl}-{tl}"]
                                df.loc[i, "f_sum_diff_pv"] = results[metric]["pivot"][acc_type]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                # masculine
                                df.loc[i, "m_cor_pv"] = results[metric]["pivot"][acc_type]["masculine"]["correct_ref"][f"{sl}-{tl}"]
                                df.loc[i, "m_wro_pv"] = results[metric]["pivot"][acc_type]["masculine"]["wrong_ref"][f"{sl}-{tl}"]
                                df.loc[i, "m_diff_pv"] = results[metric]["pivot"][acc_type]["masculine"]["diff_c_w"][f"{sl}-{tl}"]
                                df.loc[i, "m_sum_diff_pv"] = results[metric]["pivot"][acc_type]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                # additional metrics
                                df.loc[i, "f_of_cor_pv"] = results[metric]["pivot"][acc_type]["f_of_all_c"][f"{sl}-{tl}"]
                                df.loc[i, "m_of_cor_pv"] = results[metric]["pivot"][acc_type]["m_of_all_c"][f"{sl}-{tl}"]
                                df.loc[i, "diff_f_m_of_cor_pv"] = results[metric]["pivot"][acc_type]["diff_f_m_of_all_c"][f"{sl}-{tl}"]
                                df.loc[i, "gatq_pv"] = results[metric]["pivot"][acc_type]["tquality_w_gender_performance"][f"{sl}-{tl}"]

                            acc_type_2 = None
                            if acc_type == "1":
                                acc_type_2 = "2"
                            if acc_type == "female_speaker":
                                acc_type_2 = "male_speaker"

                            if acc_type_2 == None:
                                continue
                            else:
                                if f"{sl}-{tl}" in results[metric]["zero_shot"][acc_type_2]["all"]["correct_ref"]:
                                    # all
                                    df.loc[i+1, "all_cor_zs"] = results[metric]["zero_shot"][acc_type_2]["all"]["correct_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "all_wro_zs"] = results[metric]["zero_shot"][acc_type_2]["all"]["wrong_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "all_diff_zs"] = results[metric]["zero_shot"][acc_type_2]["all"]["diff_c_w"][f"{sl}-{tl}"]
                                    df.loc[i+1, "all_sum_diff_zs"] = results[metric]["zero_shot"][acc_type_2]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                    # feminine
                                    df.loc[i+1, "f_cor_zs"] = results[metric]["zero_shot"][acc_type_2]["feminine"]["correct_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "f_wro_zs"] = results[metric]["zero_shot"][acc_type_2]["feminine"]["wrong_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "f_diff_zs"] = results[metric]["zero_shot"][acc_type_2]["feminine"]["diff_c_w"][f"{sl}-{tl}"]
                                    df.loc[i+1, "f_sum_diff_zs"] = results[metric]["zero_shot"][acc_type_2]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                    # masculine
                                    df.loc[i+1, "m_cor_zs"] = results[metric]["zero_shot"][acc_type_2]["masculine"]["correct_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "m_wro_zs"] = results[metric]["zero_shot"][acc_type_2]["masculine"]["wrong_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "m_diff_zs"] = results[metric]["zero_shot"][acc_type_2]["masculine"]["diff_c_w"][f"{sl}-{tl}"]
                                    df.loc[i+1, "m_sum_diff_zs"] = results[metric]["zero_shot"][acc_type_2]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                    # additional metrics
                                    df.loc[i+1, "f_of_cor_zs"] = results[metric]["zero_shot"][acc_type_2]["f_of_all_c"][f"{sl}-{tl}"]
                                    df.loc[i+1, "m_of_cor_zs"] = results[metric]["zero_shot"][acc_type_2]["m_of_all_c"][f"{sl}-{tl}"]
                                    df.loc[i+1, "diff_f_m_of_cor_zs"] = results[metric]["zero_shot"][acc_type_2]["diff_f_m_of_all_c"][f"{sl}-{tl}"]
                                    df.loc[i+1, "gatq_zs"] = results[metric]["zero_shot"][acc_type_2]["tquality_w_gender_performance"][f"{sl}-{tl}"]
                                if f"{sl}-{tl}" in results[metric]["pivot"][acc_type_2]["all"]["correct_ref"]:
                                    # all
                                    df.loc[i+1, "all_cor_pv"] = results[metric]["pivot"][acc_type_2]["all"]["correct_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "all_wro_pv"] = results[metric]["pivot"][acc_type_2]["all"]["wrong_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "all_diff_pv"] = results[metric]["pivot"][acc_type_2]["all"]["diff_c_w"][f"{sl}-{tl}"]
                                    df.loc[i+1, "all_sum_diff_pv"] = results[metric]["pivot"][acc_type_2]["all"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                    # feminine
                                    df.loc[i+1, "f_cor_pv"] = results[metric]["pivot"][acc_type_2]["feminine"]["correct_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "f_wro_pv"] = results[metric]["pivot"][acc_type_2]["feminine"]["wrong_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "f_diff_pv"] = results[metric]["pivot"][acc_type_2]["feminine"]["diff_c_w"][f"{sl}-{tl}"]
                                    df.loc[i+1, "f_sum_diff_pv"] = results[metric]["pivot"][acc_type_2]["feminine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                    # masculine
                                    df.loc[i+1, "m_cor_pv"] = results[metric]["pivot"][acc_type_2]["masculine"]["correct_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "m_wro_pv"] = results[metric]["pivot"][acc_type_2]["masculine"]["wrong_ref"][f"{sl}-{tl}"]
                                    df.loc[i+1, "m_diff_pv"] = results[metric]["pivot"][acc_type_2]["masculine"]["diff_c_w"][f"{sl}-{tl}"]
                                    df.loc[i+1, "m_sum_diff_pv"] = results[metric]["pivot"][acc_type_2]["masculine"]["sum_c_and_diff_c_w"][f"{sl}-{tl}"]
                                    # additional metrics
                                    df.loc[i+1, "f_of_cor_pv"] = results[metric]["pivot"][acc_type_2]["f_of_all_c"][f"{sl}-{tl}"]
                                    df.loc[i+1, "m_of_cor_pv"] = results[metric]["pivot"][acc_type_2]["m_of_all_c"][f"{sl}-{tl}"]
                                    df.loc[i+1, "diff_f_m_of_cor_pv"] = results[metric]["pivot"][acc_type_2]["diff_f_m_of_all_c"][f"{sl}-{tl}"]
                                    df.loc[i+1, "gatq_pv"] = results[metric]["pivot"][acc_type_2]["tquality_w_gender_performance"][f"{sl}-{tl}"]

    # # average zero-shot directions
    # j = 12 * len(map_train_set_model_name)
    # k = num_rows + 1 + list(map_train_set_model_name.keys()).index(train_set)

    if metric == "BLEU":
        # average zero-shot directions
        d_zs = {
            "sl": "avg. zs",
            "model": map_train_set_model_name[train_set],
            "all_cor_zs": [results[metric]["zero_shot"]["all"]["correct_ref"]["zs_avg"]],
            "all_wro_zs": [results[metric]["zero_shot"]["all"]["wrong_ref"]["zs_avg"]],
            "all_diff_zs": [results[metric]["zero_shot"]["all"]["diff_c_w"]["zs_avg"]],
            "all_sum_diff_zs": [results[metric]["zero_shot"]["all"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "f_cor_zs": [results[metric]["zero_shot"]["feminine"]["correct_ref"]["zs_avg"]],
            "f_wro_zs": [results[metric]["zero_shot"]["feminine"]["wrong_ref"]["zs_avg"]],
            "f_diff_zs": [results[metric]["zero_shot"]["feminine"]["diff_c_w"]["zs_avg"]],
            "f_sum_diff_zs": [results[metric]["zero_shot"]["feminine"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "m_cor_zs": [results[metric]["zero_shot"]["masculine"]["correct_ref"]["zs_avg"]],
            "m_wro_zs": [results[metric]["zero_shot"]["masculine"]["wrong_ref"]["zs_avg"]],
            "m_diff_zs": [results[metric]["zero_shot"]["masculine"]["diff_c_w"]["zs_avg"]],
            "m_sum_diff_zs": [results[metric]["zero_shot"]["masculine"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "f_of_cor_zs": [results[metric]["zero_shot"]["f_of_all_c"]["zs_avg"]],
            "m_of_cor_zs": [results[metric]["zero_shot"]["m_of_all_c"]["zs_avg"]],
            "diff_f_m_of_cor_zs": [results[metric]["zero_shot"]["diff_f_m_of_all_c"]["zs_avg"]],
            "gatq_zs": [results[metric]["zero_shot"]["tquality_w_gender_performance"]["zs_avg"]],


            "all_cor_pv": [results[metric]["pivot"]["all"]["correct_ref"]["zs_avg"]],
            "all_wro_pv": [results[metric]["pivot"]["all"]["wrong_ref"]["zs_avg"]],
            "all_diff_pv": [results[metric]["pivot"]["all"]["diff_c_w"]["zs_avg"]],
            "all_sum_diff_pv": [results[metric]["pivot"]["all"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "f_cor_pv": [results[metric]["pivot"]["feminine"]["correct_ref"]["zs_avg"]],
            "f_wro_pv": [results[metric]["pivot"]["feminine"]["wrong_ref"]["zs_avg"]],
            "f_diff_pv": [results[metric]["pivot"]["feminine"]["diff_c_w"]["zs_avg"]],
            "f_sum_diff_pv": [results[metric]["pivot"]["feminine"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "m_cor_pv": [results[metric]["pivot"]["masculine"]["correct_ref"]["zs_avg"]],
            "m_wro_pv": [results[metric]["pivot"]["masculine"]["wrong_ref"]["zs_avg"]],
            "m_diff_pv": [results[metric]["pivot"]["masculine"]["diff_c_w"]["zs_avg"]],
            "m_sum_diff_pv": [results[metric]["pivot"]["masculine"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "f_of_cor_pv": [results[metric]["pivot"]["f_of_all_c"]["zs_avg"]],
            "m_of_cor_pv": [results[metric]["pivot"]["m_of_all_c"]["zs_avg"]],
            "diff_f_m_of_cor_pv": [results[metric]["pivot"]["diff_f_m_of_all_c"]["zs_avg"]],
            "gatq_pv": [results[metric]["pivot"]["tquality_w_gender_performance"]["zs_avg"]],
        }
        df = pd.concat([df, pd.DataFrame.from_dict(d_zs)])

        # average supervised directions
        if avg_sv and "sv_avg" in results[metric]["zero_shot"]["all"]["correct_ref"]:
            d_sv = {
                "sl": "avg. supervised",
                "model": map_train_set_model_name[train_set],
                "all_cor_zs": [results[metric]["zero_shot"]["all"]["correct_ref"]["sv_avg"]],
                "all_wro_zs": [results[metric]["zero_shot"]["all"]["wrong_ref"]["sv_avg"]],
                "all_diff_zs": [results[metric]["zero_shot"]["all"]["diff_c_w"]["sv_avg"]],
                "all_sum_diff_zs": [results[metric]["zero_shot"]["all"]["sum_c_and_diff_c_w"]["sv_avg"]],

                "f_cor_zs": [results[metric]["zero_shot"]["feminine"]["correct_ref"]["sv_avg"]],
                "f_wro_zs": [results[metric]["zero_shot"]["feminine"]["wrong_ref"]["sv_avg"]],
                "f_diff_zs": [results[metric]["zero_shot"]["feminine"]["diff_c_w"]["sv_avg"]],
                "f_sum_diff_zs": [results[metric]["zero_shot"]["feminine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                "m_cor_zs": [results[metric]["zero_shot"]["masculine"]["correct_ref"]["sv_avg"]],
                "m_wro_zs": [results[metric]["zero_shot"]["masculine"]["wrong_ref"]["sv_avg"]],
                "m_diff_zs": [results[metric]["zero_shot"]["masculine"]["diff_c_w"]["sv_avg"]],
                "m_sum_diff_zs": [results[metric]["zero_shot"]["masculine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                "f_of_cor_zs": [results[metric]["zero_shot"]["f_of_all_c"]["sv_avg"]],
                "m_of_cor_zs": [results[metric]["zero_shot"]["m_of_all_c"]["sv_avg"]],
                "diff_f_m_of_cor_zs": [results[metric]["zero_shot"]["diff_f_m_of_all_c"]["sv_avg"]],
                "gatq_zs": [results[metric]["zero_shot"]["tquality_w_gender_performance"]["sv_avg"]],

                # "all_cor_pv": [results[metric]["pivot"]["all"]["correct_ref"]["sv_avg"]],
                # "all_wro_pv": [results[metric]["pivot"]["all"]["wrong_ref"]["sv_avg"]],
                # "all_diff_pv": [results[metric]["pivot"]["all"]["diff_c_w"]["sv_avg"]],
                # "all_sum_diff_pv": [results[metric]["pivot"]["all"]["sum_c_and_diff_c_w"]["sv_avg"]],

                # "f_cor_pv": [results[metric]["pivot"]["feminine"]["correct_ref"]["sv_avg"]],
                # "f_wro_pv": [results[metric]["pivot"]["feminine"]["wrong_ref"]["sv_avg"]],
                # "f_diff_pv": [results[metric]["pivot"]["feminine"]["diff_c_w"]["sv_avg"]],
                # "f_sum_diff_pv": [results[metric]["pivot"]["feminine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                # "m_cor_pv": [results[metric]["pivot"]["masculine"]["correct_ref"]["sv_avg"]],
                # "m_wro_pv": [results[metric]["pivot"]["masculine"]["wrong_ref"]["sv_avg"]],
                # "m_diff_pv": [results[metric]["pivot"]["masculine"]["diff_c_w"]["sv_avg"]],
                # "m_sum_diff_pv": [results[metric]["pivot"]["masculine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                # "f_of_cor_pv": [results[metric]["pivot"]["f_of_all_c"]["sv_avg"]],
                # "m_of_cor_pv": [results[metric]["pivot"]["m_of_all_c"]["sv_avg"]],
                # "diff_f_m_of_cor_pv": [results[metric]["pivot"]["diff_f_m_of_all_c"]["sv_avg"]],
                # "gatq_pv": [results[metric]["pivot"]["tquality_w_gender_performance"]["sv_avg"]],

            }
            df = pd.concat([df, pd.DataFrame.from_dict(d_sv)])

    if metric == "accuracy":

        if acc_type == "1":
            category = "cat. 1"
        elif acc_type == "female_speaker":
            category = "female"
        else:
            category = ""
        
        # average zero-shot directions
        d_zs = {
            "sl": "avg. zs",
            "model": map_train_set_model_name[train_set],
            "category": category,
            "all_cor_zs": [results[metric]["zero_shot"][acc_type]["all"]["correct_ref"]["zs_avg"]],
            "all_wro_zs": [results[metric]["zero_shot"][acc_type]["all"]["wrong_ref"]["zs_avg"]],
            "all_diff_zs": [results[metric]["zero_shot"][acc_type]["all"]["diff_c_w"]["zs_avg"]],
            "all_sum_diff_zs": [results[metric]["zero_shot"][acc_type]["all"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "f_cor_zs": [results[metric]["zero_shot"][acc_type]["feminine"]["correct_ref"]["zs_avg"]],
            "f_wro_zs": [results[metric]["zero_shot"][acc_type]["feminine"]["wrong_ref"]["zs_avg"]],
            "f_diff_zs": [results[metric]["zero_shot"][acc_type]["feminine"]["diff_c_w"]["zs_avg"]],
            "f_sum_diff_zs": [results[metric]["zero_shot"][acc_type]["feminine"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "m_cor_zs": [results[metric]["zero_shot"][acc_type]["masculine"]["correct_ref"]["zs_avg"]],
            "m_wro_zs": [results[metric]["zero_shot"][acc_type]["masculine"]["wrong_ref"]["zs_avg"]],
            "m_diff_zs": [results[metric]["zero_shot"][acc_type]["masculine"]["diff_c_w"]["zs_avg"]],
            "m_sum_diff_zs": [results[metric]["zero_shot"][acc_type]["masculine"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "f_of_cor_zs": [results[metric]["zero_shot"][acc_type]["f_of_all_c"]["zs_avg"]],
            "m_of_cor_zs": [results[metric]["zero_shot"][acc_type]["m_of_all_c"]["zs_avg"]],
            "diff_f_m_of_cor_zs": [results[metric]["zero_shot"][acc_type]["diff_f_m_of_all_c"]["zs_avg"]],
            "gatq_zs": [results[metric]["zero_shot"][acc_type]["tquality_w_gender_performance"]["zs_avg"]],


            "all_cor_pv": [results[metric]["pivot"][acc_type]["all"]["correct_ref"]["zs_avg"]],
            "all_wro_pv": [results[metric]["pivot"][acc_type]["all"]["wrong_ref"]["zs_avg"]],
            "all_diff_pv": [results[metric]["pivot"][acc_type]["all"]["diff_c_w"]["zs_avg"]],
            "all_sum_diff_pv": [results[metric]["pivot"][acc_type]["all"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "f_cor_pv": [results[metric]["pivot"][acc_type]["feminine"]["correct_ref"]["zs_avg"]],
            "f_wro_pv": [results[metric]["pivot"][acc_type]["feminine"]["wrong_ref"]["zs_avg"]],
            "f_diff_pv": [results[metric]["pivot"][acc_type]["feminine"]["diff_c_w"]["zs_avg"]],
            "f_sum_diff_pv": [results[metric]["pivot"][acc_type]["feminine"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "m_cor_pv": [results[metric]["pivot"][acc_type]["masculine"]["correct_ref"]["zs_avg"]],
            "m_wro_pv": [results[metric]["pivot"][acc_type]["masculine"]["wrong_ref"]["zs_avg"]],
            "m_diff_pv": [results[metric]["pivot"][acc_type]["masculine"]["diff_c_w"]["zs_avg"]],
            "m_sum_diff_pv": [results[metric]["pivot"][acc_type]["masculine"]["sum_c_and_diff_c_w"]["zs_avg"]],

            "f_of_cor_pv": [results[metric]["pivot"][acc_type]["f_of_all_c"]["zs_avg"]],
            "m_of_cor_pv": [results[metric]["pivot"][acc_type]["m_of_all_c"]["zs_avg"]],
            "diff_f_m_of_cor_pv": [results[metric]["pivot"][acc_type]["diff_f_m_of_all_c"]["zs_avg"]],
            "gatq_pv": [results[metric]["pivot"][acc_type]["tquality_w_gender_performance"]["zs_avg"]],
        }
        df = pd.concat([df, pd.DataFrame.from_dict(d_zs)])

        ## second metric (category or speaker)
        if acc_type == "1" or acc_type == "female_speaker":
            if acc_type == "1":
                acc_type2 = "2"
                category = "cat. 2"
            else:
                acc_type2 = "male_speaker"
                category = "male"

            d2_zs = {
                "sl": "avg. zs",
                "model": map_train_set_model_name[train_set],
                "category": category,
                "all_cor_zs": [results[metric]["zero_shot"][acc_type2]["all"]["correct_ref"]["zs_avg"]],
                "all_wro_zs": [results[metric]["zero_shot"][acc_type2]["all"]["wrong_ref"]["zs_avg"]],
                "all_diff_zs": [results[metric]["zero_shot"][acc_type2]["all"]["diff_c_w"]["zs_avg"]],
                "all_sum_diff_zs": [results[metric]["zero_shot"][acc_type2]["all"]["sum_c_and_diff_c_w"]["zs_avg"]],

                "f_cor_zs": [results[metric]["zero_shot"][acc_type2]["feminine"]["correct_ref"]["zs_avg"]],
                "f_wro_zs": [results[metric]["zero_shot"][acc_type2]["feminine"]["wrong_ref"]["zs_avg"]],
                "f_diff_zs": [results[metric]["zero_shot"][acc_type2]["feminine"]["diff_c_w"]["zs_avg"]],
                "f_sum_diff_zs": [results[metric]["zero_shot"][acc_type2]["feminine"]["sum_c_and_diff_c_w"]["zs_avg"]],

                "m_cor_zs": [results[metric]["zero_shot"][acc_type2]["masculine"]["correct_ref"]["zs_avg"]],
                "m_wro_zs": [results[metric]["zero_shot"][acc_type2]["masculine"]["wrong_ref"]["zs_avg"]],
                "m_diff_zs": [results[metric]["zero_shot"][acc_type2]["masculine"]["diff_c_w"]["zs_avg"]],
                "m_sum_diff_zs": [results[metric]["zero_shot"][acc_type2]["masculine"]["sum_c_and_diff_c_w"]["zs_avg"]],

                "f_of_cor_zs": [results[metric]["zero_shot"][acc_type2]["f_of_all_c"]["zs_avg"]],
                "m_of_cor_zs": [results[metric]["zero_shot"][acc_type2]["m_of_all_c"]["zs_avg"]],
                "diff_f_m_of_cor_zs": [results[metric]["zero_shot"][acc_type2]["diff_f_m_of_all_c"]["zs_avg"]],
                "gatq_zs": [results[metric]["zero_shot"][acc_type2]["tquality_w_gender_performance"]["zs_avg"]],


                "all_cor_pv": [results[metric]["pivot"][acc_type2]["all"]["correct_ref"]["zs_avg"]],
                "all_wro_pv": [results[metric]["pivot"][acc_type2]["all"]["wrong_ref"]["zs_avg"]],
                "all_diff_pv": [results[metric]["pivot"][acc_type2]["all"]["diff_c_w"]["zs_avg"]],
                "all_sum_diff_pv": [results[metric]["pivot"][acc_type2]["all"]["sum_c_and_diff_c_w"]["zs_avg"]],

                "f_cor_pv": [results[metric]["pivot"][acc_type2]["feminine"]["correct_ref"]["zs_avg"]],
                "f_wro_pv": [results[metric]["pivot"][acc_type2]["feminine"]["wrong_ref"]["zs_avg"]],
                "f_diff_pv": [results[metric]["pivot"][acc_type2]["feminine"]["diff_c_w"]["zs_avg"]],
                "f_sum_diff_pv": [results[metric]["pivot"][acc_type2]["feminine"]["sum_c_and_diff_c_w"]["zs_avg"]],

                "m_cor_pv": [results[metric]["pivot"][acc_type2]["masculine"]["correct_ref"]["zs_avg"]],
                "m_wro_pv": [results[metric]["pivot"][acc_type2]["masculine"]["wrong_ref"]["zs_avg"]],
                "m_diff_pv": [results[metric]["pivot"][acc_type2]["masculine"]["diff_c_w"]["zs_avg"]],
                "m_sum_diff_pv": [results[metric]["pivot"][acc_type2]["masculine"]["sum_c_and_diff_c_w"]["zs_avg"]],

                "f_of_cor_pv": [results[metric]["pivot"][acc_type2]["f_of_all_c"]["zs_avg"]],
                "m_of_cor_pv": [results[metric]["pivot"][acc_type2]["m_of_all_c"]["zs_avg"]],
                "diff_f_m_of_cor_pv": [results[metric]["pivot"][acc_type2]["diff_f_m_of_all_c"]["zs_avg"]],
                "gatq_pv": [results[metric]["pivot"][acc_type2]["tquality_w_gender_performance"]["zs_avg"]],

            }
            df = pd.concat([df, pd.DataFrame.from_dict(d2_zs)])

        # average supervised directions
        if avg_sv and "sv_avg" in results[metric]["zero_shot"][acc_type]["all"]["correct_ref"]:
            d_sv = {
                "sl": "avg. supervised",
                "model": map_train_set_model_name[train_set],
                "category": category,
                "all_cor_zs": [results[metric]["zero_shot"][acc_type]["all"]["correct_ref"]["sv_avg"]],
                "all_wro_zs": [results[metric]["zero_shot"][acc_type]["all"]["wrong_ref"]["sv_avg"]],
                "all_diff_zs": [results[metric]["zero_shot"][acc_type]["all"]["diff_c_w"]["sv_avg"]],
                "all_sum_diff_zs": [results[metric]["zero_shot"][acc_type]["all"]["sum_c_and_diff_c_w"]["sv_avg"]],

                "f_cor_zs": [results[metric]["zero_shot"][acc_type]["feminine"]["correct_ref"]["sv_avg"]],
                "f_wro_zs": [results[metric]["zero_shot"][acc_type]["feminine"]["wrong_ref"]["sv_avg"]],
                "f_diff_zs": [results[metric]["zero_shot"][acc_type]["feminine"]["diff_c_w"]["sv_avg"]],
                "f_sum_diff_zs": [results[metric]["zero_shot"][acc_type]["feminine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                "m_cor_zs": [results[metric]["zero_shot"][acc_type]["masculine"]["correct_ref"]["sv_avg"]],
                "m_wro_zs": [results[metric]["zero_shot"][acc_type]["masculine"]["wrong_ref"]["sv_avg"]],
                "m_diff_zs": [results[metric]["zero_shot"][acc_type]["masculine"]["diff_c_w"]["sv_avg"]],
                "m_sum_diff_zs": [results[metric]["zero_shot"][acc_type]["masculine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                "f_of_cor_zs": [results[metric]["zero_shot"][acc_type]["f_of_all_c"]["sv_avg"]],
                "m_of_cor_zs": [results[metric]["zero_shot"][acc_type]["m_of_all_c"]["sv_avg"]],
                "diff_f_m_of_cor_zs": [results[metric]["zero_shot"][acc_type]["diff_f_m_of_all_c"]["sv_avg"]],
                "gatq_zs": [results[metric]["zero_shot"][acc_type]["tquality_w_gender_performance"]["sv_avg"]],

                # "all_cor_pv": [results[metric]["pivot"][acc_type]["all"]["correct_ref"]["sv_avg"]],
                # "all_wro_pv": [results[metric]["pivot"][acc_type]["all"]["wrong_ref"]["sv_avg"]],
                # "all_diff_pv": [results[metric]["pivot"][acc_type]["all"]["diff_c_w"]["sv_avg"]],
                # "all_sum_diff_pv": [results[metric]["pivot"][acc_type]["all"]["sum_c_and_diff_c_w"]["sv_avg"]],

                # "f_cor_pv": [results[metric]["pivot"][acc_type]["feminine"]["correct_ref"]["sv_avg"]],
                # "f_wro_pv": [results[metric]["pivot"][acc_type]["feminine"]["wrong_ref"]["sv_avg"]],
                # "f_diff_pv": [results[metric]["pivot"][acc_type]["feminine"]["diff_c_w"]["sv_avg"]],
                # "f_sum_diff_pv": [results[metric]["pivot"][acc_type]["feminine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                # "m_cor_pv": [results[metric]["pivot"][acc_type]["masculine"]["correct_ref"]["sv_avg"]],
                # "m_wro_pv": [results[metric]["pivot"][acc_type]["masculine"]["wrong_ref"]["sv_avg"]],
                # "m_diff_pv": [results[metric]["pivot"][acc_type]["masculine"]["diff_c_w"]["sv_avg"]],
                # "m_sum_diff_pv": [results[metric]["pivot"][acc_type]["masculine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                # "f_of_cor_pv": [results[metric]["pivot"][acc_type]["f_of_all_c"]["sv_avg"]],
                # "m_of_cor_pv": [results[metric]["pivot"][acc_type]["m_of_all_c"]["sv_avg"]],
                # "diff_f_m_of_cor_pv": [results[metric]["pivot"][acc_type]["diff_f_m_of_all_c"]["sv_avg"]],
                # "gatq_pv": [results[metric]["pivot"][acc_type]["tquality_w_gender_performance"]["sv_avg"]],
            }
            df = pd.concat([df, pd.DataFrame.from_dict(d_sv)])

        ## second metric (category or speaker)
        if acc_type == "1" or acc_type == "female_speaker":
            if acc_type == "1":
                acc_type2 = "2"
                category = "cat. 2"
            else:
                acc_type2 = "male_speaker"
                category = "male"
            
            if avg_sv and "sv_avg" in results[metric]["zero_shot"][acc_type2]["all"]["correct_ref"]:

                d2_sv = {
                    "sl": "avg. supervised",
                    "model": map_train_set_model_name[train_set],
                    "category": category,
                    "all_cor_zs": [results[metric]["zero_shot"][acc_type2]["all"]["correct_ref"]["sv_avg"]],
                    "all_wro_zs": [results[metric]["zero_shot"][acc_type2]["all"]["wrong_ref"]["sv_avg"]],
                    "all_diff_zs": [results[metric]["zero_shot"][acc_type2]["all"]["diff_c_w"]["sv_avg"]],
                    "all_sum_diff_zs": [results[metric]["zero_shot"][acc_type2]["all"]["sum_c_and_diff_c_w"]["sv_avg"]],

                    "f_cor_zs": [results[metric]["zero_shot"][acc_type2]["feminine"]["correct_ref"]["sv_avg"]],
                    "f_wro_zs": [results[metric]["zero_shot"][acc_type2]["feminine"]["wrong_ref"]["sv_avg"]],
                    "f_diff_zs": [results[metric]["zero_shot"][acc_type2]["feminine"]["diff_c_w"]["sv_avg"]],
                    "f_sum_diff_zs": [results[metric]["zero_shot"][acc_type2]["feminine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                    "m_cor_zs": [results[metric]["zero_shot"][acc_type2]["masculine"]["correct_ref"]["sv_avg"]],
                    "m_wro_zs": [results[metric]["zero_shot"][acc_type2]["masculine"]["wrong_ref"]["sv_avg"]],
                    "m_diff_zs": [results[metric]["zero_shot"][acc_type2]["masculine"]["diff_c_w"]["sv_avg"]],
                    "m_sum_diff_zs": [results[metric]["zero_shot"][acc_type2]["masculine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                    "f_of_cor_zs": [results[metric]["zero_shot"][acc_type2]["f_of_all_c"]["sv_avg"]],
                    "m_of_cor_zs": [results[metric]["zero_shot"][acc_type2]["m_of_all_c"]["sv_avg"]],
                    "diff_f_m_of_cor_zs": [results[metric]["zero_shot"][acc_type2]["diff_f_m_of_all_c"]["sv_avg"]],
                    "gatq_zs": [results[metric]["zero_shot"][acc_type2]["tquality_w_gender_performance"]["sv_avg"]],


                    # "all_cor_pv": [results[metric]["pivot"][acc_type2]["all"]["correct_ref"]["sv_avg"]],
                    # "all_wro_pv": [results[metric]["pivot"][acc_type2]["all"]["wrong_ref"]["sv_avg"]],
                    # "all_diff_pv": [results[metric]["pivot"][acc_type2]["all"]["diff_c_w"]["sv_avg"]],
                    # "all_sum_diff_pv": [results[metric]["pivot"][acc_type2]["all"]["sum_c_and_diff_c_w"]["sv_avg"]],

                    # "f_cor_pv": [results[metric]["pivot"][acc_type2]["feminine"]["correct_ref"]["sv_avg"]],
                    # "f_wro_pv": [results[metric]["pivot"][acc_type2]["feminine"]["wrong_ref"]["sv_avg"]],
                    # "f_diff_pv": [results[metric]["pivot"][acc_type2]["feminine"]["diff_c_w"]["sv_avg"]],
                    # "f_sum_diff_pv": [results[metric]["pivot"][acc_type2]["feminine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                    # "m_cor_pv": [results[metric]["pivot"][acc_type2]["masculine"]["correct_ref"]["sv_avg"]],
                    # "m_wro_pv": [results[metric]["pivot"][acc_type2]["masculine"]["wrong_ref"]["sv_avg"]],
                    # "m_diff_pv": [results[metric]["pivot"][acc_type2]["masculine"]["diff_c_w"]["sv_avg"]],
                    # "m_sum_diff_pv": [results[metric]["pivot"][acc_type2]["masculine"]["sum_c_and_diff_c_w"]["sv_avg"]],

                    # "f_of_cor_pv": [results[metric]["pivot"][acc_type2]["f_of_all_c"]["sv_avg"]],
                    # "m_of_cor_pv": [results[metric]["pivot"][acc_type2]["m_of_all_c"]["sv_avg"]],
                    # "diff_f_m_of_cor_pv": [results[metric]["pivot"][acc_type2]["diff_f_m_of_all_c"]["sv_avg"]],
                    # "gatq_pv": [results[metric]["pivot"][acc_type2]["tquality_w_gender_performance"]["sv_avg"]],

                }
                df = pd.concat([df, pd.DataFrame.from_dict(d2_sv)])

    df.to_csv(out_path, index=False, sep=";")
    return df

def main_mustshe():

    opt = parser.parse_args()
    raw_path = opt.raw_path
    pred_path = opt.pred_path
    train_set = opt.train_set
    out_path = opt.out_path
    
    results = get_empty_results_dict()
    results = calc_and_store_results_per_lset(results, raw_path, pred_path)
    results = calc_and_store_results_avg_zeroshot_directions(results, raw_path, pred_path)
    results = calc_and_store_results_avg_supervised_directions(results, raw_path, pred_path, train_set)

    # export results
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
        "multiwayES.SIM": "baseline_ES_AUX",
        "multiwayES.r32.q.SIM": "residual_ES_AUX",
    }

    # (2) BLEU
    out_path_bleu = f"{out_path}/summary_bleu.csv"
    if os.path.exists(out_path_bleu):
        df_bleu = pd.read_csv(out_path_bleu, sep=";")
    else:
        with open(f"{out_path}/df_bleu.pkl", "rb") as file:
            df_bleu = pickle.load(file)

    # (3) Accuracy
    out_path_acc = f"{out_path}/summary_acc.csv"
    if os.path.exists(out_path_acc):
        df_acc = pd.read_csv(out_path_acc, sep=";")
    else:
        with open(f"{out_path}/df_acc.pkl", "rb") as file:
            df_acc = pickle.load(file)

    # (4) Accuracy (cat)
    out_path_acc_cat = f"{out_path}/summary_acc_cat.csv"
    if os.path.exists(out_path_acc_cat):
        df_acc_cat = pd.read_csv(out_path_acc_cat, sep=";")
    else:
        with open(f"{out_path}/df_acc_cat.pkl", "rb") as file:
            df_acc_cat = pickle.load(file)

    # (5) Accuracy (speaker)
    out_path_acc_speaker = f"{out_path}/summary_acc_speaker.csv"
    if os.path.exists(out_path_acc_speaker):
        df_acc_speaker = pd.read_csv(out_path_acc_speaker, sep=";")
    else:
        with open(f"{out_path}/df_acc_speaker.pkl", "rb") as file:
            df_acc_speaker = pickle.load(file)

    export_results(results, "BLEU", df_bleu, out_path_bleu, map_train_set_model_name, train_set, avg_sv=False)
    export_results(results, "accuracy", df_acc, out_path_acc, map_train_set_model_name, train_set, acc_type="total", avg_sv=False)
    export_results(results, "accuracy", df_acc_cat, out_path_acc_cat, map_train_set_model_name, train_set, acc_type="1", avg_sv=False)
    export_results(results, "accuracy", df_acc_speaker, out_path_acc_speaker, map_train_set_model_name, train_set, acc_type="female_speaker", avg_sv=False)

if __name__ == "__main__":
    main_mustshe()
