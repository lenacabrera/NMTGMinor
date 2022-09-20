# models
baseline_EN="twoway.r32.q"
residual_EN="twoway.r32.q.new"
baseline_EN_AUX="twoway.SIM"
residual_EN_AUX="twoway.SIM.r32.q"
# baseline_EN_AUX="twoway.r32.q.SIM"
# residual_EN_AUX="twoway.new.SIM.r32.q"
baseline_EN_ADV="twoway.ADV"
residual_EN_ADV="twoway.ADV.r32.q"
# baseline_EN_ADV="twoway.r32.q.ADV"
# residual_EN_ADV="twoway.new.ADV.r32.q"

baseline_ES="multiwayES"
residual_ES="multiwayES.r32.q"
baseline_ES_AUX="multiwayES.SIM"
residual_ES_AUX="multiwayES.r32.q.SIM"
baseline_ES_ADV="multiwayES.ADV"
residual_ES_ADV="multiwayES.ADV.r32.q"

baseline_DE="multiwayDE"
residual_DE="multiwayDE.r32.q"

baseline_ESFRIT="multiwayESFRIT"
residual_ESFRIT="multiwayESFRIT.r32.q.new"

baseline_ES_2="twowayES"
residual_ES_2="twowayES.r32.q"
baseline_ES_AUX_2="twowayES.SIM"
residual_ES_AUX_2="twowayES.SIM.r32.q"
baseline_ES_ADV_2="twowayES.ADV"
residual_ES_ADV_2="twowayES.ADV.r32.q"

baseline_DE_2="twowayDE"
residual_DE_2="twowayDE"
baseline_DE_AUX_2="twowayDE.SIM"
residual_DE_AUX_2="twowayDE.SIM.r32.q"
baseline_DE_ADV_2="twowayDE.ADV"
residual_DE_ADV_2="twowayDE.ADV.r32.q"

baseline_EN_ADV_3="twoway.new.ADV"
baseline_ES_ADV_3="twowayES.new.ADV"
baseline_DE_ADV_3="twowayDE.new.ADV"
residual_EN_ADV_3="twoway.new.ADV.r32.q"
residual_ES_ADV_3="twowayES.new.ADV.r32.q"
residual_DE_ADV_3="twowayDE.new.ADV.r32.q"


# twoway
train_en_b="${baseline_EN} ${baseline_EN_AUX} ${baseline_EN_ADV}"
train_en_r="${residual_EN} ${residual_EN_AUX} ${residual_EN_ADV}"
train_en_3="${baseline_EN_ADV_3} ${residual_EN_ADV_3}"
train_en="${train_en_b} ${train_en_r}"

train_es_b_2="${baseline_ES_2} ${baseline_ES_AUX_2} ${baseline_ES_ADV_2}"
train_es_r_2="${residual_ES_2} ${residual_ES_AUX_2} ${residual_ES_ADV_2}"
train_es_3="${baseline_ES_ADV_3} ${residual_ES_ADV_3}"
train_es_2="${train_es_b_2} ${train_es_r_2}"

train_de_b_2="${baseline_DE_2} ${baseline_DE_AUX_2} ${baseline_DE_ADV_2}"
train_de_r_2="${residual_DE_2} ${residual_DE_AUX_2} ${residual_DE_ADV_2}"
train_de_3=" ${baseline_DE_ADV_3} ${residual_DE_ADV_3}"
train_de_2="${train_de_b_2} ${train_de_r_2}"

# # multiway
# train_es_b="${baseline_ES} ${baseline_ES_AUX} ${baseline_ES_ADV}"
# train_es_r="${residual_ES} ${residual_ES_AUX} ${residual_ES_ADV}"
# train_es="${train_es_b} ${train_es_r}"

# train_de_b="${baseline_DE} ${baseline_DE_AUX} ${baseline_DE_ADV}"
# train_de_r="${residual_DE} ${residual_DE_AUX} ${residual_DE_ADV}"
# train_de="${train_de_b} ${train_de_r}"


# group training sets
train_sets_en="${train_en} ${train_en_3}"
train_sets_es="${train_es_2} ${train_es_3}" # ${baseline_ESFRIT} ${residual_ESFRIT}"
# train_sets_es="${train_es} ${train_es_2}" # ${baseline_ESFRIT} ${residual_ESFRIT}"
train_sets_de="${train_de_2} ${train_de_3}"
# train_sets_de="${train_de} ${train_de_2}"

train_twoway_final_b_en="${baseline_EN} ${baseline_EN_AUX} ${baseline_EN_ADV_3}"
train_twoway_final_r_en="${residual_EN} ${residual_EN_AUX} ${residual_EN_ADV_3}"
train_twoway_final_en="${train_twoway_final_b_en} ${train_twoway_final_r_en}"

train_twoway_final_b_es="${baseline_ES_2} ${baseline_ES_AUX_2} ${baseline_ES_ADV_3}"
train_twoway_final_r_es="${residual_ES_2} ${residual_ES_AUX_2} ${residual_ES_ADV_3}"
train_twoway_final_es="${train_twoway_final_b_es} ${train_twoway_final_r_es}"

train_twoway_final_b_de="${baseline_DE_2} ${baseline_DE_AUX_2} ${baseline_DE_ADV_3}"
train_twoway_final_r_de="${residual_DE_2} ${residual_DE_AUX_2} ${residual_DE_ADV_3}"
train_twoway_final_de="${train_twoway_final_b_de} ${train_twoway_final_r_de}"

train_twoway_final="${train_twoway_final_en} ${train_twoway_final_es} ${train_twoway_final_de}"


train_sets="${train_twoway_final_es}"
echo $train_sets

# # mustshe
# for train_set in $train_sets; do
#     # pick correct eval set -> tokenization based on training data
#     eval_set=twoway
#     if [[ $train_sets_en == *$train_set* ]]; then
#         eval_set=twoway
#         pivot=en
#     elif [[ $train_sets_es == *$train_set* ]]; then
#         eval_set=twowayES
#         pivot=es
#     elif [[ $train_sets_de == *$train_set* ]]; then
#         eval_set=twowayDE
#         pivot=de
#     else
#         echo "Error: Unknown model"
#         exit
#     fi

#     echo $train_set $eval_set
#     # zero-shot
#     bash $SCRIPTDIR/mustshe/pred.mustshe.sh transformer.mustc $train_set $eval_set
#     # pivot
#     bash $SCRIPTDIR/mustshe/pred.pivot.mustshe.sh transformer.mustc $pivot $train_set $eval_set
# done


# mustc
for train_set in $train_sets; do
    echo $train_set
    # pick correct eval set -> tokenization based on training data
    EVAL_SET=multiway
    # if [[ $train_twoway_final_en == *$train_set* ]]; then
    #     eval_set=twoway
    #     pivot=en
    # elif [[ $train_twoway_final_es == *$train_set* ]]; then
    #     eval_set=twowayES
    #     pivot=es
    # elif [[ $train_twoway_final_de == *$train_set* ]]; then
    #     eval_set=twowayDE
    #     pivot=de
    # else
    #     echo "Error: Unknown model"
    #     exit
    # fi

    # zero-shot
    bash $SCRIPTDIR/mustc/pred.mustc.sh transformer.mustc $train_set $EVAL_SET
    echo PIVOT $train_set
    if [[ $train_sets_en == *$train_set* ]]; then
        pivot=en
    elif [[ $train_sets_es == *$train_set* ]]; then
        pivot=es
    elif [[ $train_sets_de == *$train_set* ]]; then
        pivot=de
    else
        echo "Error: Unknown model"
        exit
    fi
    bash $SCRIPTDIR/mustc/pred.pivot.mustc.sh transformer.mustc $pivot $train_set $EVAL_SET
done
