#!/bin/bash

mkdir -p export/srcnn/{2x,4x,8x}

python train.py srcnn --epochs 20 --scale 2 --patch-size 64 --batch-size 64 --save-path export/srcnn/2x
python train.py srcnn --epochs 20 --scale 4 --patch-size 64 --batch-size 64 --save-path export/srcnn/4x
python train.py srcnn --epochs 20 --scale 8 --patch-size 64 --batch-size 64 --save-path export/srcnn/8x
