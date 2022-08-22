#!/bin/bash
source ./config.sh

export systemName=mustc
export TRAIN_SET=multiwayDE # twoway.r32.q, multiwayES, multiwayDE

export BASEDIR=$WORKDIR
export LAYER=5
export TRANSFORMER=transformer
export ENC_LAYER=5
export WUS=8000
export HEAD=8

# data setup
export PREPRO_DIR=$systemName/prepro_20000_subwordnmt/$TRAIN_SET.SIM
export EPOCHS=10
export LR=2

export SKIP_TRAIN=false
export MULTILAN=true
export LAN_EMB=true
export LAN_EMB_CONCAT=true

export SKIP_PREPRO=true

export FP16=true
export MODEL=$TRANSFORMER.$PREPRO_DIR

export SIMILARITY=true
export LOAD_FROM_CKPT=model.pt
if [[ $TRAIN_SET == "twoway.r32.r" ]]; then
        # baseline_EN
        export LOAD_FROM_CKPT=model_ppl_4.960131_e64.00.pt
elif [[ $TRAIN_SET == "twoway.r32.r.new" ]]; then
        # residual_EN
        echo "Error: For 'removed residual' model run dedicated script (train.similarity.remove.residual.mustc.sh)"
elif [[ $TRAIN_SET == "multiwayES" ]]; then
        # baseline_ES
        export LOAD_FROM_CKPT=model_ppl_8.229582_e54.00.pt
elif [[ $TRAIN_SET == "multiwayES.r32.q" ]]; then
        # residual_ES
        echo "Error: For 'removed residual' model run dedicated script (train.similarity.remove.residual.mustc.sh)"
elif [[ $TRAIN_SET == "multiwayDE" ]]; then
        # # baseline_DE
        # export LOAD_FROM_CKPT=NULL
        echo "-load_from not specified yet"
        exit
elif [[ $TRAIN_SET == "multiwayDE.r32.q" ]]; then
        # residual_DE
        echo "Error: For 'removed residual' model run dedicated script (train.similarity.remove.residual.mustc.sh)"

# load_from: multiwayES -> model_ppl_8.229582_e54.00.pt , multiwayES.r32.q -> model_ppl_8.431752_e50.00.pt
# load_from: twoway.r32.q -> model_ppl_4.960131_e64.00.pt , twoway.r32.q.new -> model_ppl_5.068066_e64.00.pt
# load_from: multiwayDE -> model_ppl_9.463746_e56.00.pt , multiwayDE.r32.q -> model_ppl_9.593347_e51.00.pt

echo $MODEL

# Start training
echo 'Start training'
echo $PREPRO_DIR
echo $MODEL

mkdir $WORKDIR/model/${MODEL} -p

for f in $DATADIR/$PREPRO_DIR/binarized_mmem/; do
        ln -s -f $f $WORKDIR/model/${MODEL}/$(basename -- "$fullfile")
done

$SCRIPTDIR/mustc/utils/train.similarity.mustc.sh $PREPRO_DIR $MODEL $LOAD_FROM_CKPT

echo 'Done training'
