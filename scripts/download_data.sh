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

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip
unzip -n -j DIV2K_train_LR_x8.zip -d lr_x8

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_mild.zip
unzip -n -j DIV2K_train_LR_mild.zip -d lr_mild

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_difficult.zip
unzip -n -j DIV2K_train_LR_difficult.zip -d lr_difficult

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_wild.zip
unzip -n -j DIV2K_train_LR_wild.zip -d lr_wild

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

cd ../../

# div2k 2018 validation dataset
cd test

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
