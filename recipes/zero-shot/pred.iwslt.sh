#!/bin/bash
source ./recipes/zero-shot/config.sh
set -eu

export MODEL=$1 # model name

LAN="it nl ro en"

mkdir $DATADIR/$MODEL -p

for sl in $LAN; do
    for tl in $LAN; do
        if [[ ! "$sl" == "$tl" ]]; then

            pred_src=$DATADIR/iwslt17_multiway/test/tok/tst2017$sl-$tl.real.s # path to tokenized test data
            out=$DATADIR/$MODEL/$sl-$tl.pred

            bos='#'${tl^^}

            python3 -u $NMTDIR/translate.py -gpu $GPU \
            -model $WORKDIR/model/$MODEL/model.pt \
            -src $pred_src \
            -batch_size 128 -verbose \
            -beam_size 4 -alpha 1.0 \
            -normalize \
            -output $out \
            -fast_translate \
            -src_lang $sl \
            -tgt_lang $tl \
            -bos_token $bos
            
            # Postprocess output
            sed -e "s/@@ //g" $out  | sed -e "s/@@$//g" | sed -e "s/&apos;/'/g" -e 's/&#124;/|/g' -e "s/&amp;/&/g" -e 's/&lt;/>/g' -e 's/&gt;/>/g' -e 's/&quot;/"/g' -e 's/&#91;/[/g' -e 's/&#93;/]/g' -e 's/ - /-/g' | sed -e "s/ '/'/g" | sed -e "s/ '/'/g" | sed -e "s/%- / -/g" | sed -e "s/ -%/- /g" | perl -nle 'print ucfirst' > $out.tok
            
            $MOSESDIR/scripts/tokenizer/detokenizer.perl -l $tl < $out.tok > $out.detok
            $MOSESDIR/scripts/recaser/detruecase.perl < $out.detok > $out.pt
        
            echo '===========================================' $sl $tl
            # Evaluate against original reference  
            cat $out.pt | sacrebleu $DATADIR/iwslt17_multiway/test/orig/tst2017$tl-$sl.$tl > $OUTDIR/$MODEL/$sl-$tl.test.res
            cat $OUTDIR/$MODEL/$sl-$tl.test.res
        
        fi
    done
done
