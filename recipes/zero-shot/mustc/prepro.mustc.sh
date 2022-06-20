#!/bin/bash
export systemName=mustc
export BPESIZE=20000
export PREPRO_TYPE=${BPESIZE}
export SENTENCE_PIECE=false

if [ $SENTENCE_PIECE = true ]; then
        PREPRO_TYPE=${PREPRO_TYPE}_sentencepiece
else
        PREPRO_TYPE=${PREPRO_TYPE}_subwordnmt
fi

PREPRO_DIR=prepro_${PREPRO_TYPE}

$SCRIPTDIR/scripts/defaultPreprocessor/Train.sh $systemName $PREPRO_DIR

# # MuST-C languages
# # LAN="ar cs de en es fa fr it nl pt ro ru tr vi zh" # languages not supported by moses tokenizer: ar, fa, tr, vi, zh
LAN="cs de en es fr it nl pt ro ru" # lang. supported by moses tokenizer

for sl in $LAN; do
        for tl in $LAN; do
                if [ "$sl" != "$tl" ] && ( [ "$sl" == "en" ] || [ "$tl" == "en" ] ); then
                        echo $sl-$tl
                        export sl=$sl
                        export tl=$tl
                        $SCRIPTDIR/scripts/defaultPreprocessor/Translate.sh $sl-$tl $PREPRO_DIR $systemName
                fi
        done
done

python -u $NMTDIR/utils/add_tl_specific_token.py $DATADIR/mustc/$PREPRO_DIR/ mustc
