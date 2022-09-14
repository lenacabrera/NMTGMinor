#!/bin/bash
source ./config.sh

export systemName=mustc
export TRAIN_SET=twowayDE.new # twoway, twowayES, twowayDE

export BASEDIR=$WORKDIR
export LAYER=5
export TRANSFORMER=transformer
export ENC_LAYER=5
export WUS=400 # 400, 8000
export HEAD=8

# data setup
export PREPRO_DIR=$systemName/prepro_20000_subwordnmt/$TRAIN_SET.ADV
export EPOCHS=10
export LR=2

export SKIP_TRAIN=false
export MULTILAN=true
export LAN_EMB=true
export LAN_EMB_CONCAT=true

export SKIP_PREPRO=true

export FP16=true
export MODEL=$TRANSFORMER.$PREPRO_DIR

export ADVERSARIAL=true
export TOK_ID=3 # 3 -> en, 4 -> es

echo $MODEL

# Start training
echo 'Start training'
echo $PREPRO_DIR
echo $MODEL

mkdir $WORKDIR/model/${MODEL} -p

for f in $DATADIR/$PREPRO_DIR/binarized_mmem/*; do
        ln -s -f $f $WORKDIR/model/${MODEL}/$(basename -- "$fullfile")
done

$SCRIPTDIR/mustc/utils/train.adversarial.mustc.sh $PREPRO_DIR $MODEL

echo 'Done training'
