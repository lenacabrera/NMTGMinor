export MODEL=$1
export BASEDIR=~/export/data2/lcabrera	# path to model & orig data

mkdir $BASEDIR/data/$MODEL/pivot -p

# IWSLT languages
langs="it nl ro"

for src in $langs
do
    for tgt in $langs
    do
        if [ $src != $tgt ]; then

            echo $src " -> en -> " $tgt

                # (1) pivot into English
                export sl=$src
                export tl=en

                ln -s -f $BASEDIR/data/iwslt17_multiway/test/tok/tst2017${src}-${tgt}.real.s  $BASEDIR/data/$MODEL/pivot/tst2017${src}-en-${tgt}.real.pivotin.s # symbolic link
                
                pred_src=$BASEDIR/data/$MODEL/pivot/tst2017${src}-en-${tgt}.real.pivotin.s
                out=$BASEDIR/data/$MODEL/pivot/tst2017${src}-en-${tgt}.real.pivotin.t

                bos='#'${tl^^}

                echo "Translate to EN..."
                python3 -u $NMTDIR/translate.py -gpu $GPU \
                    -model $BASEDIR/model/$MODEL/iwslt.pt \
                    -src $pred_src \
                    -batch_size 128 -verbose \
                    -beam_size 4 -alpha 1.0 \
                    -normalize \
                    -output $out \
                    -fast_translate \
                    -src_lang $sl \
                    -tgt_lang $tl \
                    -bos_token $bos

                # (2) pivot out of English
                export sl=en
                export tl=$tgt

                ln -s -f $BASEDIR/data/$MODEL/pivot/tst2017${src}-en-${tgt}.real.pivotin.t $BASEDIR/data/$MODEL/pivot/tst2017${src}-en-${tgt}.real.pivotout.s

                pred_src=$BASEDIR/data/$MODEL/pivot/tst2017${src}-en-${tgt}.real.pivotout.s
                out=$BASEDIR/data/$MODEL/pivot/tst2017${src}-en-${tgt}.real.pivotout.t

                bos='#'${tl^^}

                echo "Translate from EN..."
                python3 -u $NMTDIR/translate.py -gpu $GPU \
                    -model $BASEDIR/model/$MODEL/iwslt.pt \
                    -src $pred_src \
                    -batch_size 128 -verbose \
                    -beam_size 4 -alpha 1.0 \
                    -normalize \
                    -output $out \
                    -fast_translate \
                    -src_lang $sl \
                    -tgt_lang $tl \
                    -bos_token $bos
    
                sed -e "s/@@ //g" $out  | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' -e 's/ - /-/g' | sed -e "s/ '/'/g" | sed -e "s/ '/'/g" | sed -e "s/%- / -/g" | sed -e "s/ -%/- /g" | perl -nle 'print ucfirst' > $out.tok

                $MOSESDIR/scripts/tokenizer/detokenizer.perl -l $tl < $out.tok > $out.detok
                $MOSESDIR/scripts/recaser/detruecase.perl < $out.detok > $out.pt
                
                rm $out.tok $out.detok
                # cat $out.pt | sacrebleu $BASEDIR/data/iwslt17_multiway/test/orig/tst2017$src-$tgt.real/tst2017$src-$tgt.real.$tgt > $out.res
                cat $out.pt | sacrebleu $BASEDIR/data/iwslt17_multiway/test/orig/tst2017$tgt-$src.$tgt > $out.res
 
        fi
    done
done
