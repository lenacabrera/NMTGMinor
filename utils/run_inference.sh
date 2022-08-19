# models
baseline_EN="twoway.r32.q"
residual_EN="twoway.r32.q.new"
baseline_EN_AUX="twoway.r32.q.SIM"
residual_EN_AUX="twoway.new.SIM.r32.q"
baseline_EN_ADV="twoway.r32.q.ADV"
residual_EN_ADV="twoway.r32.q.new.ADV"

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

train_sets_en="${baseline_EN} ${residual_EN} ${residual_EN_AUX} ${residual_EN_ADV} ${baseline_EN_ADV} ${residual_EN_ADV}"
train_sets_es="${baseline_ES} ${residual_ES} ${baseline_ES_AUX} ${residual_ES_AUX} ${baseline_ES_ADV} ${residual_ES_ADV} ${baseline_ESFRIT} ${residual_ESFRIT}"
train_sets_de="${baseline_DE} ${residual_DE}"

train_sets="${train_sets_en} ${train_sets_es} ${train_sets_de}"

# mustshe
for train_set in $train_sets; do
    echo $train_set
    # zero-shot
    bash $SCRIPTDIR/mustshe/pred.mustshe.sh transformer.mustc $train_set
    # pivot
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
    bash $SCRIPTDIR/mustshe/pred.pivot.mustshe.sh transformer.mustc $pivot $train_set
done