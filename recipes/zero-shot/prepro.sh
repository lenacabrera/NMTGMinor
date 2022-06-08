#!/bin/bash
# export systemName=pmIndia #.star.9
export systemName=iwslt17_multiway
export BPESIZE=40000
export PREPRO_TYPE=${BPESIZE}_sentencepiece
export INDIC_TOKENIZER="bash $FLORESDIR/floresv1/scripts/indic_norm_tok.sh"
# export SENTENCE_PIECE=true  # only for indic data

PREPRO_DIR=prepro_${PREPRO_TYPE}

$SCRIPTDIR/scripts/defaultPreprocessor/Train.sh $systemName $PREPRO_DIR

# # indian languages
# for sl in en te kn ml bn gu hi mr or pa; do
#         for tl in en te kn ml bn gu hi mr or pa; do
# IWSLT languages
for sl in en it nl ro; do
        for tl in en it nl ro; do
                if [ "$sl" != "$tl" ]; then
                        export sl=$sl
                        export tl=$tl
                        $SCRIPTDIR/scripts/defaultPreprocessor/Translate.sh $sl-$tl $PREPRO_DIR $systemName
                fi
        done
done
