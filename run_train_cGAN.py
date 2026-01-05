#!/bin/bash

python train_cGAN.py \
  --data #path to images
  --labels_csv #path to conditional labels
  --cond csv \
  --outdir #directory where output will be saved 
  --batch 32 --epochs 3000 --fid_every 2 --snap_every 2

