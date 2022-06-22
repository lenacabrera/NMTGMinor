mkdir -p $DATADIR/mustshe/raw/correct_ref/tmp
mkdir -p $DATADIR/mustshe/raw/wrong_ref/tmp

echo "====== Remove overlap with MuST-C training data"
python -u $NMTDIR/utils/remove_overlap_mustc_mustshe.py \
    $DATADIR/mustc/raw/train/en-cs.s \
    $DATADIR/mustc/raw/train/en-de.s \
    $DATADIR/mustc/raw/train/en-es.s \
    $DATADIR/mustc/raw/train/en-fr.s \
    $DATADIR/mustc/raw/train/en-it.s \
    $DATADIR/mustc/raw/train/en-nl.s \
    $DATADIR/mustc/raw/train/en-pt.s \
    $DATADIR/mustc/raw/train/en-ro.s \
    $DATADIR/mustc/raw/train/en-ru.s \
    $DATADIR/mustshe/raw/correct_ref/en_par.s \
    $DATADIR/mustshe/raw/correct_ref/es_par.s \
    $DATADIR/mustshe/raw/correct_ref/fr_par.s \
    $DATADIR/mustshe/raw/correct_ref/it_par.s \
    $DATADIR/mustshe/raw/wrong_ref/es_par.s \
    $DATADIR/mustshe/raw/wrong_ref/fr_par.s \
    $DATADIR/mustshe/raw/wrong_ref/it_par.s \
    $DATADIR/mustshe/raw/correct_ref/tmp/ \
    $DATADIR/mustshe/raw/wrong_ref/tmp/

echo "====== Remove overlap with MuST-C validation data"
python -u $NMTDIR/utils/remove_overlap_mustc_mustshe.py \
    $DATADIR/mustc/raw/valid/en-cs.s \
    $DATADIR/mustc/raw/valid/en-de.s \
    $DATADIR/mustc/raw/valid/en-es.s \
    $DATADIR/mustc/raw/valid/en-fr.s \
    $DATADIR/mustc/raw/valid/en-it.s \
    $DATADIR/mustc/raw/valid/en-nl.s \
    $DATADIR/mustc/raw/valid/en-pt.s \
    $DATADIR/mustc/raw/valid/en-ro.s \
    $DATADIR/mustc/raw/valid/en-ru.s \
    $DATADIR/mustshe/raw/correct_ref/tmp/en_par.s \
    $DATADIR/mustshe/raw/correct_ref/tmp/es_par.s \
    $DATADIR/mustshe/raw/correct_ref/tmp/fr_par.s \
    $DATADIR/mustshe/raw/correct_ref/tmp/it_par.s \
    $DATADIR/mustshe/raw/wrong_ref/tmp/es_par.s \
    $DATADIR/mustshe/raw/wrong_ref/tmp/fr_par.s \
    $DATADIR/mustshe/raw/wrong_ref/tmp/it_par.s \
    $DATADIR/mustshe/raw/correct_ref/ \
    $DATADIR/mustshe/raw/wrong_ref/  

rm -r $DATADIR/mustshe/raw/correct_ref/tmp/
rm -r $DATADIR/mustshe/raw/wrong_ref/tmp/

