#!/bin/bash

source config.sh

for type in train test train_only1; do
#for type in train_only1; do

    rm -rf $data_dir../$stype/"$type"_lmdb

    ./../build/tools/convert_driving_data $data_dir \
    $data_dir../$stype/$type.txt \
    $data_dir../$stype/"$type"_lmdb \
    -resize_height $height -resize_width $width -shrink 0.0
done
# -resize_height 2048 -resize_width 2048 -height 64 -width 64
