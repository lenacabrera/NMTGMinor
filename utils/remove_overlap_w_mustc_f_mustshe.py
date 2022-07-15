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

def create_add_info_dict(en, add):
    with open(en, "r") as ef:
        with open(add, "r") as addf:
            en_lines = ef.read().splitlines()
            add_lines = addf.read().splitlines()

            en_add = {}
            for l, add_l in zip(en_lines, add_lines):
                add_lines_sep = add_l.split(",")
                gender_speaker = add_lines_sep[0]
                category = add_lines_sep[1]
                gender_terms = add_lines_sep[2].split(';')
                gender_terms_cr = []
                gender_terms_wr = []
                for gt in gender_terms:
                    cr_wr = gt.split(' ')
                    gender_terms_cr.append(cr_wr[0])
                    gender_terms_wr.append(cr_wr[1])

                all_gender_terms = add_lines_sep[2].split()
                en_add[l] = {
                    "speaker_gender": gender_speaker,
                    "category": category,
                    "gender_terms_cr": gender_terms_cr,
                    "gender_terms_wr": gender_terms_wr,
                }
            return en_add

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
    en_par_mustshe, es_par_mustshe, fr_par_mustshe, it_par_mustshe, es_wr_par_mustshe, fr_wr_par_mustshe, it_wr_par_mustshe, 
    es_add_mustshe, fr_add_mustshe, it_add_mustshe, out_dir, out_dir_cr, out_dir_wr):

    en = create_list_of_en_mustc_sentences(en_cs_mustc, en_de_mustc, en_es_mustc, en_fr_mustc, en_it_mustc, en_nl_mustc, en_pt_mustc, en_ro_mustc, en_ru_mustc)
    mustshe_en_es_fr_it_d = create_language_pair_dict(en_par_mustshe, es_par_mustshe, fr_par_mustshe, it_par_mustshe)
    mustshe_en_wr_es_fr_it_d = create_language_pair_dict(en_par_mustshe, es_wr_par_mustshe, fr_wr_par_mustshe, it_wr_par_mustshe)

    es_add = create_add_info_dict(en_par_mustshe, es_add_mustshe)
    fr_add = create_add_info_dict(en_par_mustshe, fr_add_mustshe)
    it_add = create_add_info_dict(en_par_mustshe, it_add_mustshe)

    en_novl = []
    es_novl = []
    fr_novl = []
    it_novl = []
    es_wr_novl = []
    fr_wr_novl = []
    it_wr_novl = []

    ovl = []

    add_gspeaker_novl = []
    add_category_novl = []
    es_add_gterms_cr_novl = []
    es_add_gterms_wr_novl = []
    fr_add_gterms_cr_novl = []
    fr_add_gterms_wr_novl = []
    it_add_gterms_cr_novl = []
    it_add_gterms_wr_novl = []

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

            add_gspeaker_novl.append(es_add[l]["speaker_gender"])
            add_category_novl.append(es_add[l]["category"])

            es_add_gterms_cr_novl.append(es_add[l]["gender_terms_cr"])
            es_add_gterms_wr_novl.append(es_add[l]["gender_terms_wr"])
            fr_add_gterms_cr_novl.append(fr_add[l]["gender_terms_cr"])
            fr_add_gterms_wr_novl.append(fr_add[l]["gender_terms_wr"])
            it_add_gterms_cr_novl.append(it_add[l]["gender_terms_cr"])
            it_add_gterms_wr_novl.append(it_add[l]["gender_terms_wr"])

    print(len(es_add_gterms_cr_novl), len(es_add_gterms_wr_novl), len(fr_add_gterms_cr_novl))

    with open(out_dir_cr + "en_par.s", "w") as enf:
        with open(out_dir_wr + "en_par.s", "w") as enf_wr:
            print(len(en_novl), len(mustshe_en_es_fr_it_d.keys()))
            for l in en_novl:
                enf.write(l + "\n")
                enf_wr.write(l + "\n")

    with open(out_dir_cr + "es_par.s", "w") as esf:
        for l in es_novl:
            esf.write(l + "\n")

    with open(out_dir_cr + "fr_par.s", "w") as frf:
        for l in fr_novl:
            frf.write(l + "\n")

    with open(out_dir_cr + "it_par.s", "w") as itf:
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


    with open(out_dir + "speaker.csv", "w") as speakerf:
        for l in add_gspeaker_novl:
            speakerf.write(l + "\n")
    
    with open(out_dir + "category.csv", "w") as categoryf:
        for l in add_category_novl:
            categoryf.write(l + "\n")

    with open(out_dir_cr + "es_gender_terms.csv", "w") as esgtermscrf:
        for lines in es_add_gterms_cr_novl:
            for l in lines:
                esgtermscrf.write(l + " ")
            esgtermscrf.write("\n")
    with open(out_dir_cr + "it_gender_terms.csv", "w") as itgtermscrf:
        for lines in it_add_gterms_cr_novl:
            for l in lines:
                itgtermscrf.write(l + " ")
            itgtermscrf.write("\n")
    with open(out_dir_cr + "fr_gender_terms.csv", "w") as frgtermscrf:
        for lines in fr_add_gterms_cr_novl:
            for l in lines:
                frgtermscrf.write(l + " ")
            frgtermscrf.write("\n")

    with open(out_dir_wr + "es_gender_terms.csv", "w") as esgtermswrf:
        for lines in es_add_gterms_wr_novl:
            for l in lines:
                esgtermswrf.write(l + " ")
            esgtermswrf.write("\n")
    with open(out_dir_wr + "it_gender_terms.csv", "w") as itgtermswrf:
        for lines in it_add_gterms_wr_novl:
            for l in lines:
                itgtermswrf.write(l + " ")
            itgtermswrf.write("\n")
    with open(out_dir_wr + "fr_gender_terms.csv", "w") as frgtermswrf:
        for lines in fr_add_gterms_wr_novl:
            for l in lines:
                frgtermswrf.write(l + " ")
            frgtermswrf.write("\n")

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
        es_add_mustshe=args[16],
        fr_add_mustshe=args[17],
        it_add_mustshe=args[18],
        out_dir=args[19],
        out_dir_cr=args[20],
        out_dir_wr=args[21]
    )
    