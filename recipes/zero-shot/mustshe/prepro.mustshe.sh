#!/bin/bash
# bash ./recipes/zero-shot/mustshe/prepare.mustshe.sh

export systemName=mustc
export BPESIZE=20000
export PREPRO_TYPE=${BPESIZE}
export SENTENCE_PIECE=false

if [ $SENTENCE_PIECE = true ]; then
        PREPRO_TYPE=${PREPRO_TYPE}_sentencepiece
else
        PREPRO_TYPE=${PREPRO_TYPE}_subwordnmt
fi

PREPRO_DIR=prepro_${PREPRO_TYPE}

TOKDIR=$DATADIR/mustshe/$PREPRO_DIR    # path to tokenized data

mkdir -p $TOKDIR
mkdir -p $TOKDIR/correct_ref 
mkdir -p $TOKDIR/wrong_ref 

for ref in correct_ref wrong_ref; do
        for sl in en es it fr; do
                for tl in  en es it fr; do
                        if [ "$sl" != "$tl" ]; then
                                set=$sl-$tl
                                echo $set.s
                                src_file=$TOKDIR/$ref/$set.s
                                cat $DATADIR/mustshe/prep/$ref/$set.s | \
                                        perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} | \
                                        $MOSESDIR/scripts/recaser/truecase.perl --model $WORKDIR/model/$PREPRO_DIR/mustc/truecase-model.s | \
                                        $BPEDIR/subword_nmt/apply_bpe.py -c $WORKDIR/model/$PREPRO_DIR/mustc/codec --vocabulary $WORKDIR/model/$PREPRO_DIR/mustc/voc.s --vocabulary-threshold 50 \
                                                > $src_file
                                # cp -f $src_file $TOKDIR/$ref/$tl-$sl.t
                        fi
                done
        done
done
