#!/bin/bash

export MODEL=$1  # model name (e.g. baseline)
export INPUT=$2  # data set name (e.g. mustshe)
export PREPRO_TYPE=${BPESIZE}_sentencepiece
PREPRO_DIR=prepro_${PREPRO_TYPE}

if [ -z "$BPESIZE" ]; then
    BPESIZE=40000
fi

if [ -z "$MODEL" ]; then
    MODEL=baseline
fi

if [ -z "$INPUT" ]; then
    INPUT=mustshe
fi


mkdir -p $WORKDIR/data/${MODEL}/    # preprocessed data


# --- (A) TRAIN
mkdir -p $WORKDIR/tmp/${MODEL}/tok  # tokenized
mkdir -p $WORKDIR/tmp/${MODEL}/sc   # smart case

### (1) TOKENIZATION (tok)
echo "*** Learning Tokenizer" 

# Source
# echo "" > $WORKDIR/tmp/${MODEL}/corpus.tok.s
for f in $DATADIR/${INPUT}/orig/*\.s ; do
# -----> TODO: EN is done multiple times
    lan_pair="$(basename "$f")"
    sl=${lan_pair:0:2}
    echo "sl: " $sl
    # echo $f
    cat $f | perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} > $WORKDIR/tmp/${MODEL}/tok/${f##*/}
    cat $WORKDIR/tmp/${MODEL}/tok/${f##*/} >> $WORKDIR/tmp/${MODEL}/corpus.tok.s
done

# Target
# echo "" > $WORKDIR/tmp/${MODEL}/corpus.tok.t
for f in $DATADIR/${INPUT}/orig/*\.t ; do
    lan_pair="$(basename "$f")"
    tl=${lan_pair:3:2}
    echo "tl: " $tl
    # echo $f
    cat $f | perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${tl} > $WORKDIR/tmp/${MODEL}/tok/${f##*/}
    cat $WORKDIR/tmp/${MODEL}/tok/${f##*/} >> $WORKDIR/tmp/${MODEL}/corpus.tok.t
done


### (2) SMARTCASE (sc)
echo "*** Learning Truecaser" 
$MOSESDIR/scripts/recaser/train-truecaser.perl --model $WORKDIR/model/${MODEL}/truecase-model.s --corpus $WORKDIR/tmp/${MODEL}/corpus.tok.s
$MOSESDIR/scripts/recaser/train-truecaser.perl --model $WORKDIR/model/${MODEL}/truecase-model.t --corpus $WORKDIR/tmp/${MODEL}/corpus.tok.t

# Source
for f in $WORKDIR/tmp/$MODEL/tok/*\.s; do
    cat $f | \
    $MOSESDIR/scripts/recaser/truecase.perl --model $WORKDIR/model/${MODEL}/truecase-model.s > $WORKDIR/tmp/${MODEL}/sc/${f##*/}
done

# Target
for f in $WORKDIR/tmp/$MODEL/tok/*\.t; do
    cat $f | \
    $MOSESDIR/scripts/recaser/truecase.perl --model $WORKDIR/model/${MODEL}/truecase-model.t > $WORKDIR/tmp/${MODEL}/sc/${f##*/}
done

# echo "" > $WORKDIR/tmp/${MODEL}/corpus.sc.s
for f in $WORKDIR/tmp/${MODEL}/sc/*\.s; do
    cat $f >> $WORKDIR/tmp/${MODEL}/corpus.sc.s
done

# echo "" > $WORKDIR/tmp/${MODEL}/corpus.sc.t
for f in $WORKDIR/tmp/${MODEL}/sc/*\.t; do
    cat $f >> $WORKDIR/tmp/${MODEL}/corpus.sc.t
done


# ### (3) BPE
# echo "*** Learning BPE of size" $BPESIZE
# if [ ! "$SENTENCE_PIECE" == true ]; then
# 	echo "*** BPE by subword-nmt"
# 	$BPEDIR/subword_nmt/learn_joint_bpe_and_vocab.py --input $WORKDIR/tmp/${MODEL}/corpus.sc.s $WORKDIR/tmp/${MODEL}/corpus.sc.t -s $BPESIZE -o $WORKDIR/model/${MODEL}/codec --write-vocabulary $WORKDIR/model/${MODEL}/voc.s $WORKDIR/model/${MODEL}/voc.t

#     for f in $WORKDIR/tmp/${MODEL}/*\.s; do
#         # only use tokens with minimum frequency of 50 in the training set
#         $BPEDIR/subword_nmt/apply_bpe.py -c $WORKDIR/model/${MODEL}/codec --vocabulary $WORKDIR/model/${MODEL}/voc.s --vocabulary-threshold 50 < $f > $WORKDIR/data/${MODEL}/${f##*/}
#     done

# 	# for set in valid train
# 	# do
#     for f in $WORKDIR/tmp/${MODEL}/*\.t; do
#         $BPEDIR/subword_nmt/apply_bpe.py -c $WORKDIR/model/${MODEL}/codec --vocabulary $WORKDIR/model/${MODEL}/voc.t --vocabulary-threshold 50 < $f > $WORKDIR/data/${MODEL}/${f##*/}
#     done
# 	# done

# else
# 	echo "*** BPE by sentencepiece"
# 	spm_train \
# 	--input=$WORKDIR/tmp/${MODEL}/corpus.sc.s, $WORKDIR/tmp/${MODEL}/corpus.sc.t \
#         --model_prefix=$WORKDIR/data/${MODEL}/sentencepiece.bpe \
#         --vocab_size=$BPESIZE \
#         --character_coverage=1.0 \
#         --model_type=bpe
# 	# for set in valid train
# 	# do
# 		for f in $WORKDIR/tmp/${MODEL}/*; do #\.s
# 		echo $f
# 		spm_encode \
#         	--model=$WORKDIR/data/${MODEL}/sentencepiece.bpe.model \
#         	--output_format=piece \
# 		--vocabulary_threshold 50 < $f > $WORKDIR/data/${MODEL}/${f##*/}
# 		done
# 	# done
# fi

# rm -r $WORKDIR/tmp/${MODEL}/


# # # --- (A) TRANSLATE

# # mkdir -p $BASEDIR/data/${MODEL}/eval
# # # mkdir -p $BASEDIR/data/${MODEL}/valid

# # # Tokenize, Smartcase, BPE
# # cat $inFile | \
# #     perl $MOSESDIR/scripts/tokenizer/tokenizer.perl -l ${sl} | \ 
# #     $MOSESDIR/scripts/recaser/truecase.perl --model $BASEDIR/model/${MODEL}/truecase-model.s | \
# #     $BPEDIR/subword_nmt/apply_bpe.py -c $WORKDIR/model/${MODEL}/codec --vocabulary $WORKDIR/model/${MODEL}/voc.s --vocabulary-threshold 50 > $WORKDIR/data/${MODEL}/eval/$set.s

# # spm_encode \
# #     --model=$WORKDIR/data/${MODEL}/sentencepiece.bpe.model \
# #     --output_format=piece \
# #     --vocabulary_threshold 50 < $WORKDIR/data/${MODEL}/eval/$set.tok.s > $WORKDIR/data/${MODEL}/eval/$set.s

