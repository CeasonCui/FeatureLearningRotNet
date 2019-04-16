#!/bin/bash

echo downloading trained results
mkdir -p experiments
cd experiments

export fileid=1qI5znIhn_z1IvXf_9U_rigT7yC0467uh
export filename=CIFAR10_RotNet_NIN4blocks.tar
wget -q --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget -q --show-progress --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt

tar -xf CIFAR10_RotNet_NIN4blocks.tar --one-top-level
rm *.tar
cd ..
mkdir -p datasets
cd datasets

echo downloading CIFAR10 csv
mkdir -p CIFAR
cd CIFAR
export fileid=1OYAo6yGbT1EFMnX8f2loXx8qm-2BJ89x
export filename=test_batch
wget -q --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget -q --show-progress --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt

echo extracting cifar10

echo done
