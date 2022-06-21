import sys
import tqdm

def create_language_pair_dict(en, tl):
    with open(en, "r") as ef:
        with open(tl, "r") as tf:
            en_lines = ef.read().splitlines()
            tl_lines = tf.read().splitlines()
            return dict(zip(en_lines, tl_lines))

def create_list_of_en_mustc_sentences(en_cs_mustc, en_de_mustc, en_es_mustc, en_fr_mustc, en_it_mustc, en_nl_mustc, en_pt_mustc, en_ro_mustc, en_ru_mustc):
    with open(en_cs_mustc, "r") as en_cs:
        with open(en_de_mustc, "r") as en_de:
            with open(en_es_mustc, "r") as en_es:
                with open(en_fr_mustc, "r") as en_fr:
                    with open(en_it_mustc, "r") as en_it:
                        with open(en_nl_mustc, "r") as en_nl:  
                            with open(en_pt_mustc, "r") as en_pt:
                                with open(en_ro_mustc, "r") as en_ro:
                                    with open(en_ru_mustc, "r") as en_ru:
                                        en_cs_lines = en_cs.read().splitlines()
                                        en_de_lines = en_de.read().splitlines()
                                        en_es_lines = en_es.read().splitlines()
                                        en_fr_lines = en_fr.read().splitlines()
                                        en_it_lines = en_it.read().splitlines()
                                        en_nl_lines = en_nl.read().splitlines()
                                        en_pt_lines = en_pt.read().splitlines()
                                        en_ro_lines = en_ro.read().splitlines()
                                        en_ru_lines = en_ru.read().splitlines()
                                        lines = en_cs_lines
                                        lines.extend(en_de_lines)
                                        lines.extend(en_es_lines)
                                        lines.extend(en_fr_lines)
                                        lines.extend(en_it_lines)
                                        lines.extend(en_nl_lines)
                                        lines.extend(en_pt_lines)
                                        lines.extend(en_ro_lines)
                                        lines.extend(en_ru_lines)
    return lines

def check_overlap(en_cs_mustc, en_de_mustc, en_es_mustc, en_fr_mustc, en_it_mustc, en_nl_mustc, en_pt_mustc, en_ro_mustc, en_ru_mustc, 
    en_es_mustshe, es_mustshe, en_fr_mustshe, fr_mustshe, en_it_mustshe, it_mustshe, out_dir):

    en = create_list_of_en_mustc_sentences(en_cs_mustc, en_de_mustc, en_es_mustc, en_fr_mustc, en_it_mustc, en_nl_mustc, en_pt_mustc, en_ro_mustc, en_ru_mustc)
    en_es_d = create_language_pair_dict(en_es_mustshe, es_mustshe)
    en_fr_d = create_language_pair_dict(en_fr_mustshe, fr_mustshe)
    en_it_d = create_language_pair_dict(en_it_mustshe, it_mustshe)

    en_es_no_ovl = {}
    en_fr_no_ovl = {}
    en_it_no_ovl = {}

    duplicates = []
    print("Check ES...")
    for l, es in tqdm.tqdm(en_es_d.items()):
        if l in en:
            duplicates.append(l)
        else:
            en_es_no_ovl[l] = es

    print("Check FR...")
    for l, fr in tqdm.tqdm(en_fr_d.items()):
        if l in en:
            duplicates.append(l)
        else:
            en_fr_no_ovl[l] = fr

    print("Check IT...")
    for l, it in tqdm.tqdm(en_it_d.items()):
        if l in en:
            duplicates.append(l)
        else:
            en_it_no_ovl[l] = it
        
    print(f"Found {len(duplicates)} matches of {len(en_es_no_ovl) + len(en_fr_no_ovl) + len(en_it_no_ovl)} lines.")

    with open(out_dir + "en-es.s", "w") as en_es_f:
        with open(out_dir + "es-en.s", "w") as es_en_f:
            for l, es in en_es_no_ovl.items():
                en_es_f.write(l + "\n")
                es_en_f.write(es + "\n")

    with open(out_dir + "en-fr.s", "w") as en_fr_f:
        with open(out_dir + "fr-en.s", "w") as fr_en_f:
            for l, fr in en_fr_no_ovl.items():
                en_fr_f.write(l + "\n")
                fr_en_f.write(fr + "\n")

    with open(out_dir + "en-it.s", "w") as en_it_f:
        with open(out_dir + "it-en.s", "w") as it_en_f:
            for l, it in en_it_no_ovl.items():
                en_it_f.write(l + "\n")
                it_en_f.write(it + "\n")                         

    print("Done.")


if __name__ == '__main__':
    args = sys.argv[1:]

    check_overlap(
        en_cs_mustc=args[0], 
        en_de_mustc=args[1], 
        en_es_mustc=args[2],  
        en_fr_mustc=args[3], 
        en_it_mustc=args[4], 
        en_nl_mustc=args[5],
        en_pt_mustc=args[6], 
        en_ro_mustc=args[7], 
        en_ru_mustc=args[8],
        en_es_mustshe=args[9],
        es_mustshe=args[10],
        en_fr_mustshe=args[11],
        fr_mustshe=args[12],
        en_it_mustshe=args[13],
        it_mustshe=args[14],
        out_dir=args[15]
    )
