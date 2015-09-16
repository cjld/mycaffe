#!/bin/bash
source config.sh

for type in train test; do
    ./../build/tools/compute_driving_mean \
    $data_dir../$stype/"$type"_lmdb \
    $data_dir../$stype/"$type"_mean.binaryproto lmdb
done

#./../build/tools/compute_driving_mean \
#  /media/randon/LENOVO/data/tencent/pl120_train_lmdb \
#  /media/randon/LENOVO/data/tencent/pl120_train_mean.binaryproto lmdb
