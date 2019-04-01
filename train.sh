#!/usr/bin/env bash
#
# train.sh
# Copyright (C) 2018 LeonTao
#
# Distributed under terms of the MIT license.
#

export cuda_visible_devices=0
export cuda_launch_blocking=1

mkdir -p data/
mkdir -p log/
mkdir -p models/

python train.py \
    --data_dir ./data/ \
    --log log/ \
    --save_mode best \
    --save_model models/ \
    --embedding_size 128 \
    --hidden_size 256 \
    --num_layers 1 \
    --bidirectional \
    --in_channels 1 \
    --out_channels 128 \
    --kernel_heights 3 4 2 \
    --stride 1 \
    --padding 0 \
    --dropout 0.5 \
    --max_grad_norm 5.0 \
    --lr_g 0.001 \
    --lr_d 0.001 \
    --min_len 3 \
    --q_max_len 85 \
    --r_max_len 85 \
    --batch_size 128 \
    --epochs 25 \
    --device cuda \
    --es_patience 5 \
