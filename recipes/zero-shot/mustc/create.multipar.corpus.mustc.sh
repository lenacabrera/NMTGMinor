# LAN="cs de en es fr it nl pt ro ru" # lang. supported by moses tokenizer

mkdir -p $DATADIR/mustc/multipar
mkdir -p $DATADIR/mustc/multipar/train
mkdir -p $DATADIR/mustc/multipar/valid
mkdir -p $DATADIR/mustc/multipar/test

# training data
python -u $NMTDIR/utils/multi_parallel_mustc.py \
    $DATADIR/mustc/raw/train/en-cs.s \
    $DATADIR/mustc/raw/train/en-de.s \
    $DATADIR/mustc/raw/train/en-es.s \
    $DATADIR/mustc/raw/train/en-fr.s \
    $DATADIR/mustc/raw/train/en-it.s \
    $DATADIR/mustc/raw/train/en-nl.s \
    $DATADIR/mustc/raw/train/en-pt.s \
    $DATADIR/mustc/raw/train/en-ro.s \
    $DATADIR/mustc/raw/train/en-ru.s \
    $DATADIR/mustc/raw/train/en-cs.t \
    $DATADIR/mustc/raw/train/en-de.t \
    $DATADIR/mustc/raw/train/en-es.t \
    $DATADIR/mustc/raw/train/en-fr.t \
    $DATADIR/mustc/raw/train/en-it.t \
    $DATADIR/mustc/raw/train/en-nl.t \
    $DATADIR/mustc/raw/train/en-pt.t \
    $DATADIR/mustc/raw/train/en-ro.t \
    $DATADIR/mustc/raw/train/en-ru.t \
    $DATADIR/mustc/multipar/train/

# validation data
python -u $NMTDIR/utils/multi_parallel_mustc.py \
    $DATADIR/mustc/raw/valid/en-cs.s \
    $DATADIR/mustc/raw/valid/en-de.s \
    $DATADIR/mustc/raw/valid/en-es.s \
    $DATADIR/mustc/raw/valid/en-fr.s \
    $DATADIR/mustc/raw/valid/en-it.s \
    $DATADIR/mustc/raw/valid/en-nl.s \
    $DATADIR/mustc/raw/valid/en-pt.s \
    $DATADIR/mustc/raw/valid/en-ro.s \
    $DATADIR/mustc/raw/valid/en-ru.s \
    $DATADIR/mustc/raw/valid/en-cs.t \
    $DATADIR/mustc/raw/valid/en-de.t \
    $DATADIR/mustc/raw/valid/en-es.t \
    $DATADIR/mustc/raw/valid/en-fr.t \
    $DATADIR/mustc/raw/valid/en-it.t \
    $DATADIR/mustc/raw/valid/en-nl.t \
    $DATADIR/mustc/raw/valid/en-pt.t \
    $DATADIR/mustc/raw/valid/en-ro.t \
    $DATADIR/mustc/raw/valid/en-ru.t \
    $DATADIR/mustc/multipar/valid/

# test data
python -u $NMTDIR/utils/multi_parallel_mustc.py \
    $DATADIR/mustc/raw/tst-COMMON/en-cs.s \
    $DATADIR/mustc/raw/tst-COMMON/en-de.s \
    $DATADIR/mustc/raw/tst-COMMON/en-es.s \
    $DATADIR/mustc/raw/tst-COMMON/en-fr.s \
    $DATADIR/mustc/raw/tst-COMMON/en-it.s \
    $DATADIR/mustc/raw/tst-COMMON/en-nl.s \
    $DATADIR/mustc/raw/tst-COMMON/en-pt.s \
    $DATADIR/mustc/raw/tst-COMMON/en-ro.s \
    $DATADIR/mustc/raw/tst-COMMON/en-ru.s \
    $DATADIR/mustc/raw/tst-COMMON/en-cs.t \
    $DATADIR/mustc/raw/tst-COMMON/en-de.t \
    $DATADIR/mustc/raw/tst-COMMON/en-es.t \
    $DATADIR/mustc/raw/tst-COMMON/en-fr.t \
    $DATADIR/mustc/raw/tst-COMMON/en-it.t \
    $DATADIR/mustc/raw/tst-COMMON/en-nl.t \
    $DATADIR/mustc/raw/tst-COMMON/en-pt.t \
    $DATADIR/mustc/raw/tst-COMMON/en-ro.t \
    $DATADIR/mustc/raw/tst-COMMON/en-ru.t \
    $DATADIR/mustc/multipar/test/


for set in train valid test; do
    for lan in cs de es fr it nl pt ro ru; do
        cp -f $DATADIR/mustc/multipar/$set/en.s $DATADIR/mustc/multipar/$set/en-$lan.s
        cp -f $DATADIR/mustc/multipar/$set/en-$lan.s $DATADIR/mustc/multipar/$set/$lan-en.t
        cp -f $DATADIR/mustc/multipar/$set/$lan-en.s $DATADIR/mustc/multipar/$set/en-$lan.t
    done
    rm $DATADIR/mustc/multipar/$set/en.s
done
