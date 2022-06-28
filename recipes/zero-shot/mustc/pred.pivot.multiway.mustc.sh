#!/bin/bash
source ./recipes/zero-shot/config.sh

export MODEL=$1

mkdir $OUTDIR/$MODEL/mustc -p
mkdir $OUTDIR/$MODEL/mustc/pivot -p
mkdir $OUTDIR/$MODEL/mustc/pivot/multiway -p


LAN="cs de en es fr it nl pt ro ru"
pvt=en # de

for src in $LAN; do
    for tgt in $LAN; do
        echo $src-$tgt
        if [[ $src != $tgt ]] && [[ $src != en ]] && [[ $tgt != en ]]; then
        
            echo $src " -> " $pvt " -> " $tgt

            # (1) pivot into e.g. English
            export sl=$src
            export tl=$pvt

            ln -s -f $DATADIR/mustc/prepro_20000_subwordnmt/multiway/test/$sl-$tl.s  $OUTDIR/$MODEL/mustc/pivot/multiway/${src}-$pvt-${tgt}.real.pivotin.s 
            
            pred_src=$OUTDIR/$MODEL/mustc/pivot/multiway/${src}-$pvt-${tgt}.real.pivotin.s
            out=$OUTDIR/$MODEL/mustc/pivot/multiway/${src}-$pvt-${tgt}.real.pivotin.t

            bos='#'${tl^^}

            echo "Translate to pivot..."
            python3 -u $NMTDIR/translate.py \
                    -gpu $GPU \
                    -model $WORKDIR/model/$MODEL/model.pt \
                    -src $pred_src \
                    -batch_size 128 \
                    -verbose \
                    -beam_size 4 \
                    -alpha 1.0 \
                    -normalize \
                    -output $out \
                    -fast_translate \
                    -src_lang $sl \
                    -tgt_lang $tl \
                    -bos_token $bos

            # (2) pivot out of English
            export sl=$pvt
            export tl=$tgt

            ln -s -f $OUTDIR/$MODEL/mustc/pivot/multiway/${src}-$pvt-${tgt}.real.pivotin.t $OUTDIR/$MODEL/mustc/pivot/multiway/${src}-$pvt-${tgt}.real.pivotout.s

            pred_src=$OUTDIR/$MODEL/mustc/pivot/multiway/${src}-$pvt-${tgt}.real.pivotout.s
            out=$OUTDIR/$MODEL/mustc/pivot/multiway/${src}-$pvt-${tgt}.real.pivotout.t

            bos='#'${tl^^}

            echo "Translate from pivot..."
            python3 -u $NMTDIR/translate.py \
                    -gpu $GPU \
                    -model $WORKDIR/model/$MODEL/model.pt \
                    -src $pred_src \
                    -batch_size 128 \
                    -verbose \
                    -beam_size 4 \
                    -alpha 1.0 \
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

            echo '===========================================' $src $tgt
            # Evaluate against original reference  
            cat $out.pt | sacrebleu $DATADIR/mustc/raw/multiway/tst-COMMON/$src-$tgt.t > $out.res
            cat $out.res
        fi
    done
done

