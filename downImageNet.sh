#!/bin/bash

mkdir -p datasets
cd datasets
mkdir -p IMAGENET
cd IMAGENET
mkdir -p ILSVRC2012
cd ILSVRC2012
mkdir -p train
wget -c http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train_t3.tar
tar -xvf ILSVRC2012_img_train_t3.tar -C ./train/
cd train
for f in *.tar; do tar -xvf "$f" --one-top-level; done
rm *.tar
cd ..
rm *.tar
mkdir -p val/val
cd val/val
wget -c http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar
rm *.tar