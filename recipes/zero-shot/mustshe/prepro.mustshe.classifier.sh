#!/bin/bash
export systemName=mustshe
export BPESIZE=20000
export PREPRO_TYPE=${BPESIZE}
export SENTENCE_PIECE=false

TRAIN_SET=twoway

if [ $SENTENCE_PIECE = true ]; then
        PREPRO_TYPE=${PREPRO_TYPE}_sentencepiece
else
        PREPRO_TYPE=${PREPRO_TYPE}_subwordnmt
fi
PREPRO_DIR=prepro_${PREPRO_TYPE}
TOKDIR=$DATADIR/mustshe/$PREPRO_DIR/$TRAIN_SET.GEN    # path to tokenized data

mkdir -p $TOKDIR
mkdir -p $TOKDIR/train
mkdir -p $TOKDIR/train/label
mkdir -p $TOKDIR/train/label/sent
mkdir -p $TOKDIR/valid
mkdir -p $TOKDIR/valid/label
mkdir -p $TOKDIR/valid/label/sent


echo "Prepare raw data..."
bash $SCRIPTDIR/mustshe/create.classifier.training.data.sh

echo "Preprocessing with previously trained BPE..."
for dataset in train valid; do
        # for sl in en es it fr; do
        for sl in en es it fr; do
                for tl in en es it fr; do
                        # if [ "$sl" != "$tl" ]; then
                        if [[ "$sl" != "$tl" ]] && [[ "$sl" == "en" ]] || [[ "$tl" == "en" ]] ; then
                                set=$sl-$tl
                                tok_file=$TOKDIR/$dataset/$set.s
                                echo $tok_file
                                src_file=$DATADIR/mustshe/raw/$dataset/all/$set.s
                                cat $src_file | \
                                        perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} | \
                                        $MOSESDIR/scripts/recaser/truecase.perl --model $WORKDIR/model/$PREPRO_DIR/mustc/$TRAIN_SET/truecase-model.s | \
                                        $BPEDIR/subword_nmt/apply_bpe.py -c $WORKDIR/model/$PREPRO_DIR/mustc/$TRAIN_SET/codec --vocabulary $WORKDIR/model/$PREPRO_DIR/mustc/$TRAIN_SET/voc.s --vocabulary-threshold 50 \
                                                > $tok_file
                                # if [ "$tl" != "en" ]; then
                                        cp $tok_file $TOKDIR/$dataset/$tl-$sl.t
                                # fi
                        fi
                done
                if [ "$sl" != "en" ]; then
                        cp $DATADIR/mustshe/raw/$dataset/all/gen_label/sent/$sl.s $TOKDIR/$dataset/label/sent/$sl.s
                fi
        done
done

for dataset in train valid; do
        for f in $TOKDIR/$dataset/*.*; do
                lan="$(basename "$f")"
                sl=${lan:0:2} 
                if [ "$sl" == "en" ]; then
                        echo "*** remove ***" $f
                        rm $f
                fi
        done
done