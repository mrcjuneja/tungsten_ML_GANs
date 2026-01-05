#!/bin/bash

python stylegan2-ada-pytorch/train.py \
  --outdir #directory to save outputs \
  --data   #path to labeled dataset \
  --gpus 1 --batch 32 \
  --cfg stylegan2 \
  --cond=1 \
  --gamma 0 \
  --aug ada --augpipe=blit --target 0.6 \
  --kimg 20000 \
  --snap 10 \
  --metrics=fid50k_full,kid50k_full
