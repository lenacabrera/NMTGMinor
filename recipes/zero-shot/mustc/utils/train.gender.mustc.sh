#!/bin/bash

# Note these configurations:
# - shared src tgt vocabulary
# - language embedding on decoder input (to force correct language), addtive or concatenative
# - Tgt language token replaces normal BOS token

input=$1  # prepro_dir
name=$2   # model

size=512
if [ $# -ne 2 ]; then
    size=$3
fi
innersize=$((size*4))

if [ -z $LAYER ]; then
    LAYER=8
fi

if [ -z $TRANSFORMER ]; then
    TRANSFORMER=transformer
fi

if [ -z "$BASEDIR" ]; then
    BASEDIR=/
fi

if [ -z "$NMTDIR" ]; then
    NMTDIR=/opt/NMTGMinor/
fi

if [ -z "$GPU" ]; then
    GPU=0
fi

if [ $GPU -eq -1 ]; then
    gpu_string_train=""
    gpu_string_avg=""
else
    gpu_string_train="-gpus "$GPU
    gpu_string_avg="-gpu "$GPU
fi

if [ ! -z "$FP16" ]; then
    gpu_string_train=$gpu_string_train" -fp16 -fp16_mixed"
fi

echo 'GPU parameters: '$gpu_string_train

if [ -z $OPTIM ]; then
    optim_str="-optim adam -update_method noam"
elif [ $OPTIM == "noam" ]; then
    optim_str="-optim adam -update_method noam"
elif [ $OPTIM == "adam" ]; then
    optim_str="-optim adam"
else
    echo "Unkown optim methods "$OPTIM
    exit;
fi

if [ -z "$LR" ]; then
    LR=2
fi

if [ -z "$WUS" ]; then
    WUS=8000
fi

if [ -z "$EPOCHS" ]; then
    EPOCHS=128
fi

if [ -z "$HEAD" ]; then
    HEAD=8
fi

if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=3584
fi

if [ -z "$INPUT_TYPE" ]; then
    INPUT_TYPE=word
fi

if [ -z "$SKIP_TRAIN" ]; then
    SKIP_TRAIN=false
fi

if [ -z "$MULTILAN" ]; then
    MULTILAN=false
fi

if [ "$LAN_EMB" == true ]; then
    magic_str=$magic_str" -use_language_embedding"
fi

if [ "$LAN_EMB_CONCAT" == true ]; then
    magic_str=$magic_str" -language_embedding_type concat"
fi

if [ -z "$SEED" ]; then
    SEED=8877
fi

if [ -z "$DEATH" ]; then
    DEATH=0.0
fi

if [ ! -z "$RESIDUAL_AT" ]; then
    magic_str=$magic_str" -change_residual_at $RESIDUAL_AT"
fi

if [ ! -z "$RESIDUAL" ]; then
    magic_str=$magic_str" -change_residual $RESIDUAL"
fi

if [ ! -z "$QUERY_AT" ]; then
    magic_str=$magic_str" -change_att_query_at $QUERY_AT"
fi

if [ ! -z "$QUERY" ]; then
    magic_str=$magic_str" -change_att_query $QUERY"
fi

# magic_str=$magic_str" -adversarial_classifier"
magic_str=$magic_str" -gender_classifier"
magic_str=$magic_str" -gender_classifier_tok"

# echo $magic_str

mkdir -p $NMTDIR/../output/${name}
mkdir -p $BASEDIR/model/${name}/checkpoints/

DATE_AND_TIME=`date "+%Y%m%d-%H%M%S"`
echo "data in:" $BASEDIR/model/${name}/train
echo "epochs: $EPOCHS"
python3 -u $NMTDIR/train.py \
        -data $BASEDIR/model/${name}/train \
        -data_format mmem \
        -save_model $BASEDIR/model/${name}/checkpoints/model \
        -model $TRANSFORMER \
        -batch_size_words $BATCH_SIZE \
        -batch_size_update 24568 \
        -batch_size_sents 9999 \
        -batch_size_multiplier 8 \
        -checkpointing 0 \
        -layers $LAYER \
        -encoder_layers $ENC_LAYER \
        -model_size $size \
        -inner_size $innersize \
        -n_heads $HEAD \
        -label_smoothing 0.1 \
        -epochs $EPOCHS \
        $optim_str \
        -learning_rate $LR \
        -normalize_gradient \
        -tie_weights \
        -seed $SEED \
        -log_interval 1000 \
        -death_rate $DEATH \
        -join_embedding \
        -data_format mmem \
        -update_frequency -1 \
        -load_from $BASEDIR/model/${name}/checkpoints/model_ppl_4.960131_e64.00.pt \
        -gender_token_classifier $CLASSIFICATION_TYPE \
        -gradient_scale 0.1 \
        -gender_mid_layer_size 128 \
        $magic_str $gpu_string_train &> $NMTDIR/../output/${name}/${DATE_AND_TIME}_train.log
        # $magic_str $gpu_string_train &> $BASEDIR/model/${name}/{$DATE_AND_TIME}_train.log

# load_from
# twoway -> model_ppl_4.960131_e64.00.pt

cp $NMTDIR/../output/${name}/${DATE_AND_TIME}_train.log $BASEDIR/model/${name}/${DATE_AND_TIME}_train.log
checkpoints=""

# for f in `ls $BASEDIR/model/${name}/checkpoints/model_ppl_*`
# do
#     checkpoints=$checkpoints"${f}|"
# done
# checkpoints=`echo $checkpoints | sed -e "s/|$//g"`

# python3 -u $NMTDIR/average_checkpoints.py $gpu_string_avg \
#         -models $checkpoints \
#         -output $BASEDIR/model/${name}/model.pt

# rm -r $BASEDIR/tmp/${name}/