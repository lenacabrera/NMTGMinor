import sys
import tqdm

def create_language_pair_dict(en, es, fr, it):
    with open(en, "r") as ef:
        with open(es, "r") as esf:
            with open(fr, "r") as frf:
                with open(it, "r") as itf:
                    en_lines = ef.read().splitlines()
                    es_lines = esf.read().splitlines()
                    fr_lines = frf.read().splitlines()
                    it_lines = itf.read().splitlines()

                    en_es_fr_it_d = {}
                    for l, es_l, fr_l, it_l in zip(en_lines, es_lines, fr_lines, it_lines):
                        en_es_fr_it_d[l] = {
                            "es": es_l,
                            "fr": fr_l,
                            "it": it_l,
                        }
                    return en_es_fr_it_d

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
    en_par_mustshe, es_par_mustshe, fr_par_mustshe, it_par_mustshe, es_wr_par_mustshe, fr_wr_par_mustshe, it_wr_par_mustshe, out_dir, out_dir_wr):

    en = create_list_of_en_mustc_sentences(en_cs_mustc, en_de_mustc, en_es_mustc, en_fr_mustc, en_it_mustc, en_nl_mustc, en_pt_mustc, en_ro_mustc, en_ru_mustc)
    mustshe_en_es_fr_it_d = create_language_pair_dict(en_par_mustshe, es_par_mustshe, fr_par_mustshe, it_par_mustshe)
    mustshe_en_wr_es_fr_it_d = create_language_pair_dict(en_par_mustshe, es_wr_par_mustshe, fr_wr_par_mustshe, it_wr_par_mustshe)

    en_novl = []
    es_novl = []
    fr_novl = []
    it_novl = []
    es_wr_novl = []
    fr_wr_novl = []
    it_wr_novl = []

    ovl = []

    for l in tqdm.tqdm(mustshe_en_es_fr_it_d.keys()):
        if l in en:
            ovl.append(l)
        else:
            en_novl.append(l)

            es_novl.append(mustshe_en_es_fr_it_d[l]["es"])
            fr_novl.append(mustshe_en_es_fr_it_d[l]["fr"])
            it_novl.append(mustshe_en_es_fr_it_d[l]["it"]) 

            es_wr_novl.append(mustshe_en_wr_es_fr_it_d[l]["es"])
            fr_wr_novl.append(mustshe_en_wr_es_fr_it_d[l]["fr"])
            it_wr_novl.append(mustshe_en_wr_es_fr_it_d[l]["it"])  

    with open(out_dir + "en_par.s", "w") as enf:
        with open(out_dir_wr + "en_par.s", "w") as enf_wr:
            print(len(en_novl), len(mustshe_en_es_fr_it_d.keys()))
            for l in en_novl:
                enf.write(l + "\n")
                enf_wr.write(l + "\n")

    with open(out_dir + "es_par.s", "w") as esf:
        for l in es_novl:
            esf.write(l + "\n")

    with open(out_dir + "fr_par.s", "w") as frf:
        for l in fr_novl:
            frf.write(l + "\n")

    with open(out_dir + "it_par.s", "w") as itf:
        for l in it_novl:
            itf.write(l + "\n")

    with open(out_dir_wr + "es_par.s", "w") as esf:
        for l in es_wr_novl:
            esf.write(l + "\n")

    with open(out_dir_wr + "fr_par.s", "w") as frf:
        for l in fr_wr_novl:
            frf.write(l + "\n")

    with open(out_dir_wr + "it_par.s", "w") as itf:
        for l in it_wr_novl:
            itf.write(l + "\n")

    print(f"Found {len(ovl)} duplicates.")
    print(f"Remaining {len(en_novl)} sentences.")
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
        en_par_mustshe=args[9],
        es_par_mustshe=args[10],
        fr_par_mustshe=args[11],
        it_par_mustshe=args[12],
        es_wr_par_mustshe=args[13],
        fr_wr_par_mustshe=args[14],
        it_wr_par_mustshe=args[15],
        out_dir=args[16],
        out_dir_wr=args[17]
    )
    