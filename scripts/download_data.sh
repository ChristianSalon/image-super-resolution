#!/bin/bash

mkdir -p data
cd data

mkdir -p train
mkdir -p validation
mkdir -p test

cd train

# div2k 2018 training dataset
mkdir -p div2k_2018
cd div2k_2018

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip -n -j DIV2K_train_HR.zip -d hr

mkdir -p lr_nn_2x
mkdir -p lr_nn_4x
mkdir -p lr_nn_8x
mkdir -p lr_bilin_4x
mkdir -p lr_bilin_8x
mkdir -p lr_bilin_2x
mkdir -p lr_bicub_2x
mkdir -p lr_bicub_4x
mkdir -p lr_bicub_8x
mkdir -p lr_lan_2x
mkdir -p lr_lan_4x
mkdir -p lr_lan_8x

python ../../../utils/init_dataset.py --input hr --output .

cd ../../test

# div2k 2018 validation dataset
mkdir -p div2k_2018
cd div2k_2018

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip
unzip -n -j DIV2K_valid_HR.zip -d hr

mkdir -p lr_nn_2x
mkdir -p lr_nn_4x
mkdir -p lr_nn_8x
mkdir -p lr_bilin_4x
mkdir -p lr_bilin_8x
mkdir -p lr_bilin_2x
mkdir -p lr_bicub_2x
mkdir -p lr_bicub_4x
mkdir -p lr_bicub_8x
mkdir -p lr_lan_2x
mkdir -p lr_lan_4x
mkdir -p lr_lan_8x

python ../../../utils/init_dataset.py --input hr --output .

mkdir -p data/train
mkdir -p data/test

# REDS
cd data/train

mkdir -p reds
cd reds

wget -c https://seungjunnah.github.io/Datasets/reds/train_sharp.zip
unzip -n -j train_sharp.zip -d train_sharp

cd ../..

# Vid4
cd test

mkdir -p vid4
cd vid4

wget -c https://github.com/flyywh/Video-Super-Resolution-Test-Datasets/raw/master/Vid4.zip
unzip -n Vid4.zip

cd ../..
