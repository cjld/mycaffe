#!/bin/bash -x
set -e
./my/scripts/compile-nogpu.sh
./my/scripts/compile-pycaffe.sh
./my/scripts/test-pycaffe.py