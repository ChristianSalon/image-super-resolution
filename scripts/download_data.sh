#!/bin/bash

mkdir -p data
cd data

mkdir train
mkdir validation
mkdir test

cd train

# div2k 2018 training dataset
mkdir div2k_2018
cd div2k_2018

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_x8.zip
unzip -j DIV2K_train_LR_x8.zip -d lr_x8

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_mild.zip
unzip -j DIV2K_train_LR_mild.zip -d lr_mild

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_difficult.zip
unzip -j DIV2K_train_LR_difficult.zip -d lr_difficult

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_wild.zip
unzip -j DIV2K_train_LR_wild.zip -d lr_wild

wget -c http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip -j DIV2K_train_HR.zip -d hr
