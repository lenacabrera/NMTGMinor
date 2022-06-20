import os
import sys
import shutil


# def prepend_tokens(PREPRO_DIR, TLAN):
#     # PREPRO_DIR = "../../../../export/data2/lcabrera/data/prepro_20000_sentencepiece/"
#     extended_TLAN = TLAN[:]
#     extended_TLAN.append("en")
#     print(extended_TLAN)

#     if not os.path.exists(PREPRO_DIR + "/../bos/"):
#         os.mkdir(PREPRO_DIR + "/../bos/")

#     for set in ["train", "valid", "test"]:
#         path = PREPRO_DIR + set + "/"
#         path_bos = PREPRO_DIR + "/../bos/" + set + "/"

#         if not os.path.exists(path_bos):
#             os.mkdir(path_bos)

#         for tl in TLAN:
#             print(tl)

#             if set != "test":
#                 # target files
#                 file = "en-" + tl + ".t"
#                 with open(path + file) as fp:
#                     lines = fp.read().splitlines()
#                 with open(path_bos + file, "w") as fp:
#                     for line in lines:
#                         fp.write("#" + tl.upper() + " " + line + "\n")

#                 file = tl + "-en" + ".t"
#                 with open(path + file) as fp:
#                     lines = fp.read().splitlines()
#                 with open(path_bos + file, "w") as fp:
#                     for line in lines:
#                         fp.write("#" + 'en'.upper() + " " + line + "\n")

#                 # source files
#                 file = "en-" + tl + ".s"
#                 # with open(path + file) as fp:
#                 #     lines = fp.read().splitlines()
#                 with open(path_bos + file, "w") as fp:
#                     for line in lines:
#                         fp.write(line + "\n")
#                         # fp.write("#" + tl.upper() + " " + line + "\n")

#                 file = tl + "-en" + ".s"
#                 # with open(path + file) as fp:
#                 #     lines = fp.read().splitlines()
#                 with open(path_bos + file, "w") as fp:
#                     for line in lines:
#                         fp.write(line + "\n")
#                         # fp.write("#" + 'en'.upper() + " " + line + "\n")

#             elif set == "test":
#                 for sl in extended_TLAN:
#                     if sl != tl:
#                         # source files
#                         file = sl + "-" + tl + ".s"
#                         # with open(path + file) as fp:
#                         #     lines = fp.read().splitlines()
#                         with open(path_bos + file, "w") as fp:
#                             for line in lines:
#                                 fp.write(line + "\n")
#                                 # fp.write("#" + sl.upper() + " " + line + "\n")

#                         file = tl + "-" + sl + ".s"
#                         # with open(path + file) as fp:
#                         #     lines = fp.read().splitlines()
#                         with open(path_bos + file, "w") as fp:
#                             for line in lines:
#                                 fp.write(line + "\n")
#                                 # fp.write("#" + tl.upper() + " " + line + "\n")


#             # # source
#             # file = "en-" + tl + ".s"
#             # print(tl)
#             # with open(path_bos + file, "w") as fp:
#             #     for line in lines:
#             #         fp.write(line + "\n")

#             # file = tl + "-en" + ".s"
#             # with open(path_bos + file, "w") as fp:
#             #     for line in lines:
#             #         fp.write(line + "\n")


def prepend_tokens_iwslt(PREPRO_DIR):
    TLAN = ["it", "nl", "ro"]
    extended_TLAN = TLAN[:]
    extended_TLAN.append("en")
    print(extended_TLAN)

    if not os.path.exists(PREPRO_DIR + "/bos/"):
        os.mkdir(PREPRO_DIR + "/bos/")

    for set in ["train", "valid", "test"]:
        path = PREPRO_DIR + set + "/"
        path_bos = PREPRO_DIR + "/bos/" + set + "/"

        if not os.path.exists(path_bos):
            os.mkdir(path_bos)

        for tl in TLAN:
            print(tl)

            if set != "test":
                # target files
                file = "en-" + tl + ".t"
                with open(path + file) as fp:
                    lines = fp.read().splitlines()
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        # add target-language specific bos token
                        fp.write("#" + tl.upper() + " " + line + "\n")

                file = tl + "-en" + ".t"
                with open(path + file) as fp:
                    lines = fp.read().splitlines()
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        # add target-language specific bos token
                        fp.write("#" + 'en'.upper() + " " + line + "\n")

                # source files
                file = "en-" + tl + ".s"
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        fp.write(line + "\n")

                file = tl + "-en" + ".s"
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        fp.write(line + "\n")

            elif set == "test":
                for sl in extended_TLAN:
                    if sl != tl:
                        # source files
                        file = sl + "-" + tl + ".s"
                        with open(path_bos + file, "w") as fp:
                            for line in lines:
                                fp.write(line + "\n")

                        file = tl + "-" + sl + ".s"
                        with open(path_bos + file, "w") as fp:
                            for line in lines:
                                fp.write(line + "\n")


def prepend_tokens_mustc(PREPRO_DIR):
    TLAN = ["cs", "de", "es", "fr", "it", "nl", "pt", "ro", "ru"]
    extended_TLAN = TLAN[:]
    extended_TLAN.append("en")
    print(extended_TLAN)

    if not os.path.exists(PREPRO_DIR + "/bos/"):
        os.mkdir(PREPRO_DIR + "/bos/")

    if not os.path.exists(PREPRO_DIR + "/no_bos/"):
        os.mkdir(PREPRO_DIR + "/no_bos/")

    for set in ["train", "valid", "test"]:
        path = PREPRO_DIR + set + "/"
        path_bos = PREPRO_DIR + "/bos/" + set + "/"

        if not os.path.exists(path_bos):
            os.mkdir(path_bos)

        for tl in TLAN:
            print(tl)

            if set != "test":
                # target files
                file = "en-" + tl + ".t"
                print(file)
                with open(path + file) as fp:
                    lines = fp.read().splitlines()
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        # add target-language specific bos token
                        fp.write("#" + tl.upper() + " " + line + "\n")

                file = tl + "-en" + ".t"
                print(file)
                with open(path + file) as fp:
                    lines = fp.read().splitlines()
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        # add target-language specific bos token
                        fp.write("#" + 'en'.upper() + " " + line + "\n")

                # source files
                file = "en-" + tl + ".s"
                print(file)
                with open(path + file) as fp:
                    lines = fp.read().splitlines()
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        fp.write(line + "\n")

                file = tl + "-en" + ".s"
                print(file)
                with open(path + file) as fp:
                    lines = fp.read().splitlines()
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        fp.write(line + "\n")

            elif set == "test":
                file = "en-" + tl + ".s"
                # source files
                print(file)
                with open(path + file) as fp:
                    lines = fp.read().splitlines()
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        fp.write(line + "\n")

                file = tl + "-en" + ".s"
                print(file)
                with open(path + file) as fp:
                    lines = fp.read().splitlines()
                with open(path_bos + file, "w") as fp:
                    for line in lines:
                        fp.write(line + "\n")

    shutil.move(PREPRO_DIR + "/train", PREPRO_DIR + "/no_bos/train")   
    shutil.move(PREPRO_DIR + "/valid", PREPRO_DIR + "/no_bos/valid")   
    shutil.move(PREPRO_DIR + "/test", PREPRO_DIR + "/no_bos/test")  

    shutil.move(PREPRO_DIR + "bos/train", PREPRO_DIR + "/train")   
    shutil.move(PREPRO_DIR + "bos/valid", PREPRO_DIR + "/valid")   
    shutil.move(PREPRO_DIR + "bos/test", PREPRO_DIR + "/test")   

    os.rmdir(PREPRO_DIR + "/bos")


if __name__ == '__main__':
    args = sys.argv[1:]
    prepro_dir = args[0]
    input = args[1]
    if "iwslt" in input:
        prepend_tokens_iwslt(prepro_dir)
    elif "mustc" in input:
        prepend_tokens_mustc(prepro_dir)
    else:
        "no target languages found... check prepro dir (argv[1][0])"

    # tlan = ["it", "nl", "ro"]
    # tlan = ["cs", "de", "es", "fr", "it", "nl", "pt", "ro", "ru"]
    # prepend_tokens(prepro_dir, tlan)
  