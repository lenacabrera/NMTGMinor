import os
import csv

PREPRO_DIR = "../../../../export/data2/lcabrera/data/prepro_20000_sentencepiece/"

if not os.path.exists(PREPRO_DIR + "bos/"):
    os.mkdir(PREPRO_DIR + "bos/")

for set in ["train", "valid"]:
    # path = "./raw/" + set + "/"
    path = PREPRO_DIR + set + "/"
    # path_bos = "./raw/bos/" + set + "/"
    path_bos = PREPRO_DIR + "bos/" + set + "/"

    if not os.path.exists(path_bos):
        os.mkdir(path_bos)

    for tl in ["it", "nl", "ro"]:
        file = "en-" + tl + ".t"
        with open(path + file) as fp:
            lines = fp.read().splitlines()
        with open(path_bos + file, "w") as fp:
            for line in lines:
                fp.write("#" + tl.upper() + " " + line + "\n")

        file = tl + "-en" + ".t"
        tl = 'en'
        with open(path + file) as fp:
            lines = fp.read().splitlines()
        with open(path_bos + file, "w") as fp:
            for line in lines:
                fp.write("#" + tl.upper() + " " + line + "\n")

        # copy source files (as is)
        file = "en-" + tl + ".s"
        with open(path_bos + file, "w") as fp:
            for line in lines:
                fp.write(line + "\n")

        file = tl + "-en" + ".s"
        with open(path_bos + file, "w") as fp:
            for line in lines:
                fp.write(line + "\n")
