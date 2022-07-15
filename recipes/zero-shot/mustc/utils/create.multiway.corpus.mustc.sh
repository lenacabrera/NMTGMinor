# LAN="cs de en es fr it nl pt ro ru" # lang. supported by moses tokenizer

mkdir -p $DATADIR/mustc/raw/multiway
mkdir -p $DATADIR/mustc/raw/multiway/train
mkdir -p $DATADIR/mustc/raw/multiway/valid
mkdir -p $DATADIR/mustc/raw/multiway/tst-COMMON
mkdir -p $DATADIR/mustc/raw/multiway/tst-HE

echo "====== Training data"
python -u $NMTDIR/utils/multi_parallel_mustc.py \
    $DATADIR/mustc/raw/twoway/train/en-cs.s \
    $DATADIR/mustc/raw/twoway/train/en-de.s \
    $DATADIR/mustc/raw/twoway/train/en-es.s \
    $DATADIR/mustc/raw/twoway/train/en-fr.s \
    $DATADIR/mustc/raw/twoway/train/en-it.s \
    $DATADIR/mustc/raw/twoway/train/en-nl.s \
    $DATADIR/mustc/raw/twoway/train/en-pt.s \
    $DATADIR/mustc/raw/twoway/train/en-ro.s \
    $DATADIR/mustc/raw/twoway/train/en-ru.s \
    $DATADIR/mustc/raw/twoway/train/en-cs.t \
    $DATADIR/mustc/raw/twoway/train/en-de.t \
    $DATADIR/mustc/raw/twoway/train/en-es.t \
    $DATADIR/mustc/raw/twoway/train/en-fr.t \
    $DATADIR/mustc/raw/twoway/train/en-it.t \
    $DATADIR/mustc/raw/twoway/train/en-nl.t \
    $DATADIR/mustc/raw/twoway/train/en-pt.t \
    $DATADIR/mustc/raw/twoway/train/en-ro.t \
    $DATADIR/mustc/raw/twoway/train/en-ru.t \
    $DATADIR/mustc/raw/multiway/train/

echo "====== Validation data"
python -u $NMTDIR/utils/multi_parallel_mustc.py \
    $DATADIR/mustc/raw/twoway/valid/en-cs.s \
    $DATADIR/mustc/raw/twoway/valid/en-de.s \
    $DATADIR/mustc/raw/twoway/valid/en-es.s \
    $DATADIR/mustc/raw/twoway/valid/en-fr.s \
    $DATADIR/mustc/raw/twoway/valid/en-it.s \
    $DATADIR/mustc/raw/twoway/valid/en-nl.s \
    $DATADIR/mustc/raw/twoway/valid/en-pt.s \
    $DATADIR/mustc/raw/twoway/valid/en-ro.s \
    $DATADIR/mustc/raw/twoway/valid/en-ru.s \
    $DATADIR/mustc/raw/twoway/valid/en-cs.t \
    $DATADIR/mustc/raw/twoway/valid/en-de.t \
    $DATADIR/mustc/raw/twoway/valid/en-es.t \
    $DATADIR/mustc/raw/twoway/valid/en-fr.t \
    $DATADIR/mustc/raw/twoway/valid/en-it.t \
    $DATADIR/mustc/raw/twoway/valid/en-nl.t \
    $DATADIR/mustc/raw/twoway/valid/en-pt.t \
    $DATADIR/mustc/raw/twoway/valid/en-ro.t \
    $DATADIR/mustc/raw/twoway/valid/en-ru.t \
    $DATADIR/mustc/raw/multiway/valid/

echo "====== Test data (tst-COMMON) -> required for pivoting"
python -u $NMTDIR/utils/multi_parallel_mustc.py \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-cs.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-de.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-es.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-fr.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-it.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-nl.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-pt.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-ro.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-ru.s \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-cs.t \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-de.t \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-es.t \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-fr.t \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-it.t \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-nl.t \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-pt.t \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-ro.t \
    $DATADIR/mustc/raw/twoway/tst-COMMON/en-ru.t \
    $DATADIR/mustc/raw/multiway/tst-COMMON/

echo "====== Test data (tst-HE) -> required for pivoting"
python -u $NMTDIR/utils/multi_parallel_mustc.py \
    $DATADIR/mustc/raw/twoway/tst-HE/en-cs.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-de.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-es.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-fr.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-it.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-nl.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-pt.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-ro.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-ru.s \
    $DATADIR/mustc/raw/twoway/tst-HE/en-cs.t \
    $DATADIR/mustc/raw/twoway/tst-HE/en-de.t \
    $DATADIR/mustc/raw/twoway/tst-HE/en-es.t \
    $DATADIR/mustc/raw/twoway/tst-HE/en-fr.t \
    $DATADIR/mustc/raw/twoway/tst-HE/en-it.t \
    $DATADIR/mustc/raw/twoway/tst-HE/en-nl.t \
    $DATADIR/mustc/raw/twoway/tst-HE/en-pt.t \
    $DATADIR/mustc/raw/twoway/tst-HE/en-ro.t \
    $DATADIR/mustc/raw/twoway/tst-HE/en-ru.t \
    $DATADIR/mustc/raw/multiway/tst-HE/


sets="train valid tst-COMMON tst-HE"
for set in $sets; do
    for f in $DATADIR/mustc/raw/multiway/$set/*\.s; do
        lan="$(basename "$f")"
        sl=${lan:0:2}
        for tl in cs de en es fr it nl pt ro ru; do
            if [ "$sl" != "$tl" ]; then
                cp -f $f $DATADIR/mustc/raw/multiway/$set/$sl-$tl.s
            fi
        done
        rm $f
    done
    for f in $DATADIR/mustc/raw/multiway/$set/*\.s; do
        lan="$(basename "$f")"
        sl=${lan:0:2}
        tl=${lan:3:2}
        cp -f $f $DATADIR/mustc/raw/multiway/$set/$tl-$sl.t
    done
done
