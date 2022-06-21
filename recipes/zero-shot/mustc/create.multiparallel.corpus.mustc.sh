# LAN="cs de en es fr it nl pt ro ru" # lang. supported by moses tokenizer

mkdir -p $DATADIR/mustc/multipar

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
    $DATADIR/mustc/multipar/
