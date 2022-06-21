
mkdir -p $DATADIR/mustshe/no_overlap/
mkdir -p $DATADIR/mustshe/no_overlap/correct_ref
mkdir -p $DATADIR/mustshe/no_overlap/correct_ref/tmp
mkdir -p $DATADIR/mustshe/no_overlap/wrong_ref
mkdir -p $DATADIR/mustshe/no_overlap/wrong_ref/tmp

# correct ref
echo "Check MuST-C training data against MuST-SHE correct reference"
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
    $DATADIR/mustshe/orig/correct_ref/en-es.s \
    $DATADIR/mustshe/orig/correct_ref/es-en.s \
    $DATADIR/mustshe/orig/correct_ref/en-fr.s \
    $DATADIR/mustshe/orig/correct_ref/fr-en.s \
    $DATADIR/mustshe/orig/correct_ref/en-it.s \
    $DATADIR/mustshe/orig/correct_ref/it-en.s \
    $DATADIR/mustshe/no_overlap/correct_ref/tmp/

echo "Check MuST-C validation data against MuST-SHE correct reference"
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
    $DATADIR/mustshe/no_overlap/correct_ref/tmp/en-es.s \
    $DATADIR/mustshe/no_overlap/correct_ref/tmp/es-en.s \
    $DATADIR/mustshe/no_overlap/correct_ref/tmp/en-fr.s \
    $DATADIR/mustshe/no_overlap/correct_ref/tmp/fr-en.s \
    $DATADIR/mustshe/no_overlap/correct_ref/tmp/en-it.s \
    $DATADIR/mustshe/no_overlap/correct_ref/tmp/it-en.s \
    $DATADIR/mustshe/no_overlap/correct_ref/  

rm -r $DATADIR/mustshe/no_overlap/correct_ref/tmp

# wrong ref -> actually, no need to check because wrong ref eq. correct ref arficially modified
echo "Check MuST-C training data against MuST-SHE wrong reference"
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
    $DATADIR/mustshe/orig/wrong_ref/en-es.s \
    $DATADIR/mustshe/orig/wrong_ref/es-en.s \
    $DATADIR/mustshe/orig/wrong_ref/en-fr.s \
    $DATADIR/mustshe/orig/wrong_ref/fr-en.s \
    $DATADIR/mustshe/orig/wrong_ref/en-it.s \
    $DATADIR/mustshe/orig/wrong_ref/it-en.s \
    $DATADIR/mustshe/no_overlap/wrong_ref/tmp/

echo "Check MuST-C validation data against MuST-SHE wrong reference"
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
    $DATADIR/mustshe/no_overlap/wrong_ref/tmp/en-es.s \
    $DATADIR/mustshe/no_overlap/wrong_ref/tmp/es-en.s \
    $DATADIR/mustshe/no_overlap/wrong_ref/tmp/en-fr.s \
    $DATADIR/mustshe/no_overlap/wrong_ref/tmp/fr-en.s \
    $DATADIR/mustshe/no_overlap/wrong_ref/tmp/en-it.s \
    $DATADIR/mustshe/no_overlap/wrong_ref/tmp/it-en.s \
    $DATADIR/mustshe/no_overlap/wrong_ref/  

rm -r $DATADIR/mustshe/no_overlap/wrong_ref/tmp
