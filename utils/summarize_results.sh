#!/bin/bash
MODEL=transformer.mustc
# TEST_SET=mustshe
TEST_SET=mustc
# TRAIN_SET=twoway.r32.q.new
TRAIN_SET=multiwayDE

out_path=$NMTDIR/../output/results_auto
out_path_pkl=$NMTDIR/../output/results_auto/pkl
out_path_json=$NMTDIR/../output/results_auto/json
out_path_csv=$NMTDIR/../output/results_auto/csv

mkdir -p $out_path
mkdir -p $out_path_pkl
mkdir -p $out_path_json
mkdir -p $out_path_csv


python3 -u $NMTDIR/utils/create_results_data_frames.py \
        -in_path $out_path/xlsx \
        -out_path $out_path_pkl

rm -f $out_path_csv/summary_bleu.csv
rm -f $out_path_csv/summary_acc.csv
rm -f $out_path_csv/summary_acc_cat.csv
rm -f $out_path_csv/summary_acc_speaker.csv


baseline_EN="twoway.r32.q"
residual_EN="twoway.r32.q.new"
baseline_EN_AUX="twoway.SIM"
residual_EN_AUX="twoway.SIM.r32.q"
baseline_EN_ADV="twoway.ADV"
residual_EN_ADV="twoway.ADV.r32.q"
# baseline_EN_ADV="twoway.r32.q.ADV"
# residual_EN_ADV="twoway.new.ADV.r32.q"
# residual_EN_ADV="twoway.r32.q.new.ADV"

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
residual_DE_2="twowayDE.r32.q"
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

# train_sets_en="${baseline_EN} ${residual_EN} ${baseline_EN_AUX} ${residual_EN_AUX} ${baseline_EN_ADV} ${residual_EN_ADV}"
# train_sets_es="${baseline_ES_2} ${residual_ES_2} ${baseline_ES_AUX_2} ${residual_ES_AUX_2} ${baseline_ES_ADV_2} ${residual_ES_ADV_2}"
# train_sets_de="${baseline_DE_2} ${residual_DE_2} ${baseline_DE_AUX_2} ${residual_DE_AUX_2} ${baseline_DE_ADV_2} ${residual_DE_ADV_2}"
train_sets_en="${baseline_EN} ${residual_EN} ${baseline_EN_AUX} ${residual_EN_AUX} ${baseline_EN_ADV_3} ${residual_EN_ADV_3}"
train_sets_es="${baseline_ES_2} ${residual_ES_2} ${baseline_ES_AUX_2} ${residual_ES_AUX_2} ${baseline_ES_ADV_3} ${residual_ES_ADV_3}"
train_sets_de="${baseline_DE_2} ${residual_DE_2} ${baseline_DE_AUX_2} ${residual_DE_AUX_2} ${baseline_DE_ADV_3} ${residual_DE_ADV_3}"
train_sets="${train_sets_en} ${train_sets_es} ${train_sets_de}"
# train_sets="${baseline_EN_ADV_3} ${baseline_ES_ADV_3} ${baseline_DE_ADV_3} ${residual_EN_ADV_3} ${residual_ES_ADV_3} ${residual_DE_ADV_3}"

# train_sets="twoway.r32.q twoway.r32.q.new multiwayES multiwayES.r32.q multiwayDE multiwayDE.r32.q multiwayESFRIT multiwayESFRIT.r32.q.new multiwayES.SIM multiwayES.r32.q.SIM multiwayES.ADV multiwayES.ADV.r32.q twoway.r32.q.SIM twoway.new.SIM.r32.q twoway.r32.q.ADV twoway.r32.q.new.ADV twowayES twowayDE"
# train_sets="twowayES twowayES.r32.q"

# mustshe
for train_set in $train_sets; do
    echo $train_set
    python3 -u $NMTDIR/utils/prep_results_new.py \
            -raw_path $DATADIR/mustshe/raw \
            -pred_path $OUTDIR/$MODEL/mustshe/$train_set \
            -train_set $train_set \
            -df_path $out_path_pkl \
            -out_path_json $out_path_json \
            -out_path_csv $out_path_csv \
            -out_path $out_path
done



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


train_sets="${train_twoway_final_en}"
TEST_SET="mustc"
# train_sets="multiwayES.SIM multiwayES.r32.q.SIM multiwayES.ADV multiwayES.ADV.r32.q" # multiwayES.ADV.en" # multiwayES.ADV.en.r32.q"
# train_sets="multiwayES.SIM"
for train_set in $train_sets; do
    path=$OUTDIR/$MODEL/$TEST_SET/$train_set/
    path_pivot=$OUTDIR/$MODEL/$TEST_SET/$train_set/pivot
    echo $train_set
    # direct
    out_exp=/home/lperez/output/$train_set/$TEST_SET/direct
    rm -fr $out_exp
    mkdir -p $out_exp
    for f in $path/*\.pt; do
            cp $f $out_exp
    done
    # pivot
    out_exp_piv=/home/lperez/output/$train_set/$TEST_SET/pivot
    rm -fr $out_exp_piv
    mkdir -p $out_exp_piv
    for f in $path_pivot/*\.pt; do
            cp $f $out_exp_piv
    done
        
    if [[ $TEST_SET == "mustc" ]]; then
        # direct
        out=/home/lperez/output/results_${TEST_SET}_${train_set}.csv
        rm -f $out
        for f in $path/*\.res; do
            while read -r line; do 
                if [[ $line == "\"score"* ]]; then
                    filename="$(basename "$f")"
                    [[ ${filename} =~ [a-z][a-z]-[a-z][a-z] ]] && set=$BASH_REMATCH
                    [[ ${line:9:4} =~ ^.[0-9]*.[0-9]* ]] && score=$BASH_REMATCH
                    echo "$set;$score" >> $out
                fi
            done < $f
        done
        # pivot
        out_piv=/home/lperez/output/results_${TEST_SET}_${train_set}_pivot.csv
        rm -f $out_piv
        for f in $path_pivot/*\.res; do
            while read -r line; do 
                if [[ $line == "\"score"* ]]; then
                    filename="$(basename "$f")"
                    [[ ${filename} =~ [a-z][a-z]-[a-z][a-z]-[a-z][a-z] ]] && set=$BASH_REMATCH
                    [[ ${line:9:4} =~ ^.[0-9]*.[0-9]* ]] && score=$BASH_REMATCH
                    echo "$set;$score" >> $out_piv
                fi
            done < $f
        done
    fi
done





# if [[ $TEST_SET == "mustc" ]]; then
#     # direct
#     out=/home/lperez/output/results_${TEST_SET}_${train_set}.csv
#     rm -f $out
#     for f in $path/*\.res; do
#         while read -r line; do 
#             if [[ $line == "\"score"* ]]; then
#                 filename="$(basename "$f")"
#                 [[ ${filename} =~ [a-z][a-z]-[a-z][a-z] ]] && set=$BASH_REMATCH
#                 [[ ${line:9:4} =~ ^.[0-9]*.[0-9]* ]] && score=$BASH_REMATCH
#                 echo "$set;$score" >> $out
#             fi
#         done < $f
#     done
#     # pivot
#     out_piv=/home/lperez/output/results_${TEST_SET}_${train_set}_pivot.csv
#     rm -f $out_piv
#     for f in $path_pivot/*\.res; do
#         while read -r line; do 
#             if [[ $line == "\"score"* ]]; then
#                 filename="$(basename "$f")"
#                 [[ ${filename} =~ [a-z][a-z]-[a-z][a-z]-[a-z][a-z] ]] && set=$BASH_REMATCH
#                 [[ ${line:9:4} =~ ^.[0-9]*.[0-9]* ]] && score=$BASH_REMATCH
#                 echo "$set;$score" >> $out_piv
#             fi
#         done < $f
#     done
# fi


# if [[ $TEST_SET == "mustshe" ]]; then
#     for ref in correct_ref wrong_ref; do  
#         # direct
#         out=/home/lperez/output/results_${TEST_SET}_${TRAIN_SET}_${ref}.csv
#         rm -f $out
#         for f in $path/$ref/*\.res; do
#             while read -r line; do 
#                 if [[ $line == "\"score"* ]]; then
#                     filename="$(basename "$f")"
#                     [[ ${filename} =~ [a-z][a-z]-[a-z][a-z] ]] && set=$BASH_REMATCH
#                     [[ ${line:9:4} =~ ^.[0-9]*.[0-9]* ]] && score=$BASH_REMATCH
#                     echo "$set;$score" >> $out
#                 fi
#             done < $f
#         done
#         # pivot
#         out_piv=/home/lperez/output/results_${TEST_SET}_${TRAIN_SET}_${ref}_pivot.csv
#         rm -f $out_piv
#         for f in $path_pivot/$ref/*\.res; do
#             while read -r line; do 
#                 if [[ $line == "\"score"* ]]; then
#                     filename="$(basename "$f")"
#                     [[ ${filename} =~ [a-z][a-z]-[a-z][a-z]-[a-z][a-z] ]] && set=$BASH_REMATCH
#                     [[ ${line:9:4} =~ ^.[0-9]*.[0-9]* ]] && score=$BASH_REMATCH
#                     echo "$set;$score" >> $out_piv
#                 fi
#             done < $f
#         done
#     done
# fi


# ###################################################################
# # export preds
# if [[ $TEST_SET == "mustshe" ]]; then
#     for ref in correct_ref wrong_ref; do
#         # direct
#         out_exp=/home/lperez/output/$TRAIN_SET/$TEST_SET/direct
#         rm -fr $out_exp/$ref
#         mkdir -p $out_exp/$ref
#         for f in $path/$ref/*\.pt; do
#             cp $f $out_exp/$ref
#         done
#         # pivot
#         out_exp_piv=/home/lperez/output/$TRAIN_SET/$TEST_SET/pivot
#         rm -fr $out_exp_piv/$ref
#         mkdir -p $out_exp_piv/$ref
#         for f in $path_pivot/$ref/*\.pt; do
#             cp $f $out_exp_piv/$ref
#         done
#     done
# fi
        
# if [[ $TEST_SET == "mustc" ]]; then
#     # direct
#     out_exp=/home/lperez/output/$TRAIN_SET/$TEST_SET/direct
#     rm -fr $out_exp
#     mkdir -p $out_exp
#     for f in $path/*\.pt; do
#         cp $f $out_exp
#     done
#     # pivot
#     out_exp_piv=/home/lperez/output/$TRAIN_SET/$TEST_SET/pivot
#     rm -fr $out_exp_piv
#     mkdir -p $out_exp_piv
#     for f in $path_pivot/*\.pt; do
#         cp $f $out_exp_piv
#     done
# fi