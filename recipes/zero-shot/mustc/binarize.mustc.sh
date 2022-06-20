name=baseline
input=mustc/prepro_20000_subwordnmt

# S_LAN="en|en|en|en|en|en|en|en|en|cs|de|es|fr|it|nl|pt|ro|ru" # has to be sorted alphabetically
# T_LAN="cs|de|es|fr|it|nl|pt|ro|ru|en|en|en|en|en|en|en|en|en"

S_LAN="cs|de|en|en|en|en|en|en|en|en|en|es|fr|it|nl|pt|ro|ru" # has to be sorted alphabetically
T_LAN="en|en|cs|de|es|fr|it|nl|pt|ro|ru|en|en|en|en|en|en|en"

IFS='|' read -r -a arrayS <<< $S_LAN
IFS='|' read -r -a arrayT <<< $T_LAN

BASEDIR=$WORKDIR
echo $BASEDIR

rm -r $BASEDIR/tmp/${name}
mkdir $BASEDIR/tmp/${name} -p

datadir=$BASEDIR/data/$input/binarized_mmem/train
mkdir $datadir -p

# for each language pair, e.g. (hi, en), (ne, en)
for l in s t
do
    for set in train valid
    do
        # loop through language pairs
        for index in "${!arrayS[@]}"  #pair in te-en ta-en #ne-en si-en
        do
                pair="${arrayS[index]}-${arrayT[index]}"
                echo -n "" > $BASEDIR/tmp/${name}/$set-$pair.$l

                for f in $BASEDIR/data/${input}/$set/$pair*\.${l}
                do  # write out to tmp folder
                        cat $f >> $BASEDIR/tmp/${name}/$set-$pair.$l
                done
        done
    done
done

# concat with "|" as delimitter
function join_by { local IFS="$1"; shift; echo "$*"; }

python3 $NMTDIR/preprocess.py \
       -train_src `join_by '|' $BASEDIR/tmp/${name}/train*\.s` \
       -train_tgt `join_by '|' $BASEDIR/tmp/${name}/train*\.t` \
       -valid_src `join_by '|' $BASEDIR/tmp/${name}/valid*\.s` \
       -valid_tgt `join_by '|' $BASEDIR/tmp/${name}/valid*\.t` \
       -train_src_lang $S_LAN \
       -train_tgt_lang $T_LAN \
       -valid_src_lang $S_LAN \
       -valid_tgt_lang $T_LAN \
       -save_data $datadir \
       -src_seq_length 512 \
       -tgt_seq_length 512 \
       -join_vocab \
       -no_bos \
       -num_threads 16 \
       -format mmem
        