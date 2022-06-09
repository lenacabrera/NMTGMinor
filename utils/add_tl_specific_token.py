import os
import csv

if not os.path.exists("./prepro_40000_sentencepiece/bos/"):
    os.mkdir("./prepro_40000_sentencepiece/bos/")

for set in ["train", "valid"]:
    # path = "./raw/" + set + "/"
    path = "./prepro_40000_sentencepiece/" + set + "/"
    # path_bos = "./raw/bos/" + set + "/"
    path_bos = "./prepro_40000_sentencepiece/bos/" + set + "/"

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
