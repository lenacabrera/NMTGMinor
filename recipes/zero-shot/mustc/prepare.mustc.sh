#!/bin/bash
source ./recipes/zero-shot/config.sh

mkdir -p $DATADIR/mustc/raw/

for lp_dir in $DATADIR/mustc/orig_download/*-* ; do
    lan="$(basename "$lp_dir")"
    echo $lan
    sl=${lan:0:2}
    tl=${lan:3:2}
    if [[ "ar fa tr vi zh" != *$tl* ]]; then  # exclude languages not supported by moses tokenizer
        for set_dir in $lp_dir/*; do
            set="$(basename "$set_dir")"
            mkdir -p $DATADIR/mustc/raw/$set
            for f in $set_dir/*.*; do
                cp -f $set_dir/$set.$sl $DATADIR/mustc/raw/$set/$sl-$tl.s
                cp -f $set_dir/$set.$tl $DATADIR/mustc/raw/$set/$sl-$tl.t 
            done
        done
    fi
done

for set_dir in $DATADIR/mustc/raw/*; do
    echo $set_dir
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
cd $DATADIR/mustc/raw/
find . -depth -type d -name dev -execdir mv {} valid \;
