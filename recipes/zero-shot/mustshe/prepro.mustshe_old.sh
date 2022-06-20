#!/bin/bash
bash ./recipes/zero-shot/mustshe/prepare.mustshe.sh

if [ -z "$BPESIZE" ]; then
    BPESIZE=40000
fi

if [ -z "$MODEL" ]; then
    MODEL=baseline
fi

if [ -z "$INPUT" ]; then
    INPUT=mustshe
fi

if [ -z "$origname" ]; then
    origname=mustshe
fi

export PREPRO_TYPE=${BPESIZE}_sentencepiece
PREPRO_DIR=prepro_${PREPRO_TYPE}
TOKDIR=$DATADIR/mustshe/tok    # path to tokenized data

mkdir -p $TOKDIR 

# MuST-SHE languages: it es fr en
for sl in en es it fr; do
        for tl in  en es it fr; do
                if [ "$sl" != "$tl" ]; then
                        export sl=$sl
                        export tl=$tl
                        
                        # source
                        set=$sl-$tl
                        echo $set.s
                        # use learned BPE model
                        cat $DATADIR/mustshe/tmp/$set.s | \
                            perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} | \
                            $MOSESDIR/scripts/recaser/truecase.perl --model $WORKDIR/model/$PREPRO_DIR/truecase-model.s | \
                            $BPEDIR/subword_nmt/apply_bpe.py -c $WORKDIR/model/$PREPRO_DIR/codec --vocabulary $WORKDIR/model/$PREPRO_DIR/voc.s --vocabulary-threshold 50 \
                                        > $TOKDIR/$set.s

                fi
        done
done

# target
for f in $TOKDIR/*\.s; do
    lan="$(basename "$f")"
    sl=${lan:0:2}
    # echo $f
    for tl in en it fr es; do
        if [ "$sl" != "$tl" ]; then
            cp -f $f  $TOKDIR/$tl-$sl.t
        fi
    done
done

rm -r $DATADIR/mustshe/tmp