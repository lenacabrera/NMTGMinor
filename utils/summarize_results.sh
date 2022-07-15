#!/bin/bash

# path=$OUTDIR/transformer.mustc/mustc/multiwayES.r32.q
path=$OUTDIR/transformer.mustc/mustshe/multiwayES/pivot/correct_ref
# echo $path

# path=$1
pivot=true

# out=/home/lperez/output/results_mustc_multiwayES.r32.q.csv
out=/home/lperez/output/results_mustshe_multiwayES_corref_pivot.csv

rm -f $out

if [[ $pivot != true ]]; then
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
fi

if [[ $pivot == true ]]; then
    for f in $path/*\.res; do
        while read -r line; do 
            if [[ $line == "\"score"* ]]; then
                filename="$(basename "$f")"
                [[ ${filename} =~ [a-z][a-z]-[a-z][a-z]-[a-z][a-z] ]] && set=$BASH_REMATCH
                [[ ${line:9:4} =~ ^.[0-9]*.[0-9]* ]] && score=$BASH_REMATCH
                echo "$set;$score" >> $out
            fi
        done < $f
    done
fi
