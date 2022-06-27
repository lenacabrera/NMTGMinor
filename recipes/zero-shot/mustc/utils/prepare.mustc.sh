#!/bin/bash
source ./recipes/zero-shot/config.sh

REMOVE_OVERLAP_W_MUSTSHE=$1  # true

echo "** Create 2-way MuST-C corpus"
rm -rf $DATADIR/mustc/raw/twoway
mkdir -p $DATADIR/mustc/raw/twoway
for lp_dir in $DATADIR/mustc/raw/download/*-* ; do
    lan="$(basename "$lp_dir")"
    sl=${lan:0:2}
    tl=${lan:3:2}
    if [[ "ar fa tr vi zh" != *$tl* ]]; then  # exclude languages not supported by moses tokenizer
        for set_dir in $lp_dir/*; do
            set="$(basename "$set_dir")"
            mkdir -p $DATADIR/mustc/raw/twoway/$set
            for f in $set_dir/*.*; do
                cp -f $set_dir/$set.$sl $DATADIR/mustc/raw/twoway/$set/$sl-$tl.s
                cp -f $set_dir/$set.$tl $DATADIR/mustc/raw/twoway/$set/$sl-$tl.t 
            done
        done
    fi
done

for set_dir in $DATADIR/mustc/raw/twoway/*; do
    for f in $set_dir/*\.s; do
        lan="$(basename "$f")"
        sl=${lan:0:2}
        tl=${lan:3:2}
        cp -f $f $set_dir/$tl-$sl.t
        echo $tl-$sl.t
    done
    for f in $set_dir/*\.t; do
        lan="$(basename "$f")"
        tl=${lan:0:2}
        sl=${lan:3:2}
        cp -f $f $set_dir/$sl-$tl.s
        echo $sl-$tl.s
    done
done

# rename dev -> valid
cd $DATADIR/mustc/raw/twoway
find . -depth -type d -name dev -execdir mv {} valid \;

if [[ $REMOVE_OVERLAP_W_MUSTSHE == true ]]; then
    echo "** Remove sentences overlaping with MuST-SHE data"
    bash $SCRIPTDIR/mustc/utils/remove.overlap.mustshe.mustc.sh
fi

echo "** Create multi-way MuST-C corpus"
$SCRIPTDIR/mustc/utils/create.multiway.corpus.mustc.sh