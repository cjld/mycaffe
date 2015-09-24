#!/bin/bash -x

set -e

echo "Copy config file"

cfile=Makefile.config
cp Makefile.config.example $cfile

echo "Change config file options"
sed 's/# CPU_ONLY := 1/CPU_ONLY := 1/' $cfile -i

make clean
make all test -j4
make runtest
