path=$DATADIR/mustc/prepro_20000_subwordnmt/multiway

out_path_ES=$DATADIR/mustc/prepro_20000_subwordnmt/multiwayES
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayES
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayES/train
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayES/valid

out_path_FR=$DATADIR/mustc/prepro_20000_subwordnmt/multiwayFR
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayFR
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayFR/train
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayFR/valid

out_path_IT=$DATADIR/mustc/prepro_20000_subwordnmt/multiwayIT
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayIT
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayIT/train
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiwayIT/valid

out_path_es_it_fr=$DATADIR/mustc/prepro_20000_subwordnmt/multiway_es_it_fr
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiway_es_it_fr
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiway_es_it_fr/train
mkdir -p $DATADIR/mustc/prepro_20000_subwordnmt/multiway_es_it_fr/valid

 for set in train valid; do

     # es
     for f in $path/$set/es-*\.s; do
         cp -f $f $out_path_ES/$set
     done

     for f in $path/$set/*-es*\.s; do
         cp -f $f $out_path_ES/$set
     done

     for f in $path/$set/es-*\.t; do
         cp -f $f $out_path_ES/$set
     done

     for f in $path/$set/*-es*\.t; do
         cp -f $f $out_path_ES/$set
     done

     # fr
     for f in $path/$set/fr-*\.s; do
         cp -f $f $out_path_FR/$set
     done

     for f in $path/$set/*-fr*\.s; do
         cp -f $f $out_path_FR/$set
     done

     for f in $path/$set/fr-*\.t; do
         cp -f $f $out_path_FR/$set
     done

     for f in $path/$set/*-fr*\.t; do
         cp -f $f $out_path_FR/$set
     done

     # it
     for f in $path/$set/it-*\.s; do
         cp -f $f $out_path_IT/$set
     done

     for f in $path/$set/*-it*\.s; do
         cp -f $f $out_path_IT/$set
     done

     for f in $path/$set/it-*\.t; do
         cp -f $f $out_path_IT/$set
     done

     for f in $path/$set/*-it*\.t; do
         cp -f $f $out_path_IT/$set
     done
 done

partial_path=$DATADIR/mustc/prepro_20000_subwordnmt
for lan in multiwayES multiwayFR multiwayIT; do
    for set in train valid; do
        for f in $partial_path/$lan/$set/*; do
            echo $lan $set
            echo $f
            cp -f $f $out_path_es_it_fr/$set
        done
    done
done
