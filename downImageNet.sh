#!/bin/bash

echo downloading trained results
mkdir -p experiments
cd experiments

export fileid=175gPpsmgpzIC-TiYPrMUrfRdNXq-v67Y
export filename=ImageNet_RotNet_AlexNet.tar
wget -q --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget -q --show-progress --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt

tar -xf ImageNet_RotNet_AlexNet.tar --one-top-level

# cifar10
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

echo downloading places205 csv
mkdir -p Places205
cd Places205
wget -q --show-progress -c http://data.csail.mit.edu/places/places205/trainvalsplit_places205.tar.gz
echo extracting places205 csv
tar -xzf trainvalsplit_places205.tar.gz
rm *.gz
cd ..

echo downloading imagenet 2012 training data part 3
mkdir -p IMAGENET
cd IMAGENET
mkdir -p ILSVRC2012
cd ILSVRC2012
mkdir -p train

export fileid=1LcW1qBPS77vZju_kKCBpaH1Wb3zwrQ0n
export filename=ILSVRC2012_img_train_t3.tar
wget -q --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget -q --show-progress --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt

# wget -c http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train_t3.tar
echo extracting imagenet 2012 training data part 3
tar -xf ILSVRC2012_img_train_t3.tar -C ./train/
cd train
for f in *.tar; do tar -xf "$f" --one-top-level; done
rm *.tar
cd ..
rm *.tar

echo downloading imagenet 2012 validation data
mkdir -p val/val
cd val/val

export fileid=13NQpUJWXIkcV9xkFo-lr9X7CSmraG2Ll
export filename=ILSVRC2012_img_val.tar
wget -q --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget -q --show-progress --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt
# wget -c http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

echo extracting imagenet 2012 validation data
tar -xf ILSVRC2012_img_val.tar
rm *.tar

echo done
