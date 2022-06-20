#!/bin/bash
source ./recipes/zero-shot/config.sh

mkdir -p $DATADIR/mustshe/prep
mkdir -p $DATADIR/mustshe/prep/correct_ref
mkdir -p $DATADIR/mustshe/prep/wrong_ref
mkdir -p $DATADIR/mustshe/orig/
mkdir -p $DATADIR/mustshe/orig/correct_ref
mkdir -p $DATADIR/mustshe/orig/wrong_ref

# add mode to args
python3 -u $NMTDIR/utils/prepro_tsv.py $DATADIR/mustshe/

for ref in correct_ref wrong_ref; do
    for f in $DATADIR/mustshe/prep/$ref/*\.s; do
        lan="$(basename "$f")"
        sl=${lan:0:2}
        # echo $f
        for tl in en es it fr; do
            if [ "$sl" != "$tl" ]; then
                # file prepared for tokenization 
                cp -f $f  $DATADIR/mustshe/prep/$ref/$sl-$tl.s
                # orig data for evaluation
                cp -f $f  $DATADIR/mustshe/orig/$ref/$sl-$tl.s
                cp -f $f  $DATADIR/mustshe/orig/$ref/$tl-$sl.t
            fi
        done
    done
done
