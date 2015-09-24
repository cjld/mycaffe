#!/bin/bash -e

set -e

if cat /proc/version | grep ubuntu; then
    sudo apt-get install gfortran
else
    sudo yum install gcc-gfortran
fi

echo "Install python req libs"
cd python
for req in $(cat requirements.txt); do sudo pip install $req; done
