#!/bin/bash

sudo pip install -r requirements.txt
wget https://gist.githubusercontent.com/SykieChen/62fe1c15dfc864af862e8fd4c7b2eed3/raw/a37f73ca64efa68e09ee254054651bdb21bddaca/functional.py
sudo mv functional.py /usr/local/lib/python2.7/dist-packages/torchvision/transforms/functional.py

mkdir -p experiments
cd experiments

export fileid=175gPpsmgpzIC-TiYPrMUrfRdNXq-v67Y
export filename=ImageNet_RotNet_AlexNet.tar
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt

tar -xvf ImageNet_RotNet_AlexNet.tar --one-top-level
rm *.tar
cd ..
mkdir -p datasets
cd datasets
mkdir -p IMAGENET
cd IMAGENET
mkdir -p ILSVRC2012
cd ILSVRC2012
mkdir -p train

export fileid=1LcW1qBPS77vZju_kKCBpaH1Wb3zwrQ0n
export filename=ILSVRC2012_img_train_t3.tar
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt

# wget -c http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train_t3.tar
tar -xvf ILSVRC2012_img_train_t3.tar -C ./train/
cd train
for f in *.tar; do tar -xvf "$f" --one-top-level; done
rm *.tar
cd ..
rm *.tar
mkdir -p val/val
cd val/val

export fileid=13NQpUJWXIkcV9xkFo-lr9X7CSmraG2Ll
export filename=ILSVRC2012_img_val.tar
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O $filename \
     'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
rm -f confirm.txt cookies.txt
# wget -c http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

tar -xvf ILSVRC2012_img_val.tar
rm *.tar
