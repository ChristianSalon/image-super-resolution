New-Item -ItemType Directory -Force -Path "export/srcnn/2x"
New-Item -ItemType Directory -Force -Path "export/srcnn/4x"
New-Item -ItemType Directory -Force -Path "export/srcnn/8x"

python train.py srcnn --epochs 20 --scale 2 --patch-size 64 --batch-size 64 --save-path export/srcnn/2x
python train.py srcnn --epochs 20 --scale 4 --patch-size 64 --batch-size 64 --save-path export/srcnn/4x
python train.py srcnn --epochs 20 --scale 8 --patch-size 64 --batch-size 64 --save-path export/srcnn/8x
