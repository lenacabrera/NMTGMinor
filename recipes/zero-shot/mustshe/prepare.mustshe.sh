#!/bin/bash
source ./recipes/zero-shot/config.sh

REMOVE_OVERLAP_W_MUSTC=$1

mkdir -p $DATADIR/mustshe/raw
mkdir -p $DATADIR/mustshe/raw/correct_ref
mkdir -p $DATADIR/mustshe/raw/wrong_ref

# TODO add mode to args
python3 -u $NMTDIR/utils/prepro_tsv_mustshe.py $DATADIR/mustshe/raw/ correct_ref wrong_ref

if [[ $REMOVE_OVERLAP_W_MUSTC == true ]]; then
    echo "Remove sentences overlaping with MuST-C data"
    bash $SCRIPTDIR/mustshe/remove.overlap.mustc.mustshe.sh
fi

for ref in correct_ref wrong_ref; do
    for f in $DATADIR/mustshe/raw/$ref/*\_par.s; do
        lan="$(basename "$f")"
        sl=${lan:0:2}
        for tl in en es fr it; do
            if [ "$sl" != "$tl" ]; then
                cp $f $DATADIR/mustshe/raw/$ref/$sl-$tl.s
            fi
        done
        rm $f
    done
    for f in $DATADIR/mustshe/raw/$ref/*\.s; do
        lan="$(basename "$f")"
        sl=${lan:0:2}
        tl=${lan:3:2}
        cp $f $DATADIR/mustshe/raw/$ref/$tl-$sl.t
    done
done
