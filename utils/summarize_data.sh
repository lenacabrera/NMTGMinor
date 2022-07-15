# path=$1
# path=$DATADIR/mustshe/prepro_20000_subwordnmt/correct_ref
path=$DATADIR/mustc/prepro_20000_subwordnmt/twoway/valid
out=/home/lperez/output/summary_mustc_twoway_valid.csv

rm -f $out

for f in $path/*; do
    filename="$(basename "$f")"
    linecount=$(wc -l < $f)
    echo "$filename;$linecount" >> $out
done
