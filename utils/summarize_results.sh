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

# train_sets="multiwayESFRIT"
train_sets="twoway.r32.q twoway.r32.q.new multiwayES multiwayES.r32.q multiwayDE multiwayDE.r32.q multiwayESFRIT multiwayESFRIT.r32.q.new multiwayES.SIM multiwayES.r32.q.SIM multiwayES.ADV multiwayES.ADV.r32.q twoway.r32.q.SIM twoway.new.SIM.r32.q twoway.r32.q.ADV twoway.r32.q.new.ADV"

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

# TEST_SET="mustc"
# train_sets="multiwayES.SIM multiwayES.r32.q.SIM multiwayES.ADV multiwayES.ADV.r32.q" # multiwayES.ADV.en" # multiwayES.ADV.en.r32.q"
# # train_sets="multiwayES.SIM"
# for train_set in $train_sets; do
#     path=$OUTDIR/$MODEL/$TEST_SET/$train_set/
#     path_pivot=$OUTDIR/$MODEL/$TEST_SET/$train_set/pivot
#     echo $train_set
#     # direct
#     out_exp=/home/lperez/output/$train_set/$TEST_SET/direct
#     rm -fr $out_exp
#     mkdir -p $out_exp
#     for f in $path/*\.pt; do
#             cp $f $out_exp
#     done
#     # pivot
#     out_exp_piv=/home/lperez/output/$train_set/$TEST_SET/pivot
#     rm -fr $out_exp_piv
#     mkdir -p $out_exp_piv
#     for f in $path_pivot/*\.pt; do
#             cp $f $out_exp_piv
#     done
        
#     if [[ $TEST_SET == "mustc" ]]; then
#         # direct
#         out=/home/lperez/output/results_${TEST_SET}_${train_set}.csv
#         rm -f $out
#         for f in $path/*\.res; do
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
#         out_piv=/home/lperez/output/results_${TEST_SET}_${train_set}_pivot.csv
#         rm -f $out_piv
#         for f in $path_pivot/*\.res; do
#             while read -r line; do 
#                 if [[ $line == "\"score"* ]]; then
#                     filename="$(basename "$f")"
#                     [[ ${filename} =~ [a-z][a-z]-[a-z][a-z]-[a-z][a-z] ]] && set=$BASH_REMATCH
#                     [[ ${line:9:4} =~ ^.[0-9]*.[0-9]* ]] && score=$BASH_REMATCH
#                     echo "$set;$score" >> $out_piv
#                 fi
#             done < $f
#         done
#     fi
# done





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