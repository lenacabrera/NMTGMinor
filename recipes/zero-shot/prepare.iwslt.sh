# format test files into correct file format
for sl in de en it nl ro; do
    for tl in de en it nl ro; do  
        if [ "$sl" != "$tl" ]; then
            for f in $DATADIR/iwslt17_multiway/raw/test/*\.${sl}; do
                cp -f $f  $DATADIR/iwslt17_multiway/raw/test/$sl-$tl.s
                cp -f $f  $DATADIR/iwslt17_multiway/raw/test/$tl-$sl.t
            done
            # rm $f
        fi
    done
done