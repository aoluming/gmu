#!/usr/bin/env bash

DATASET_NAME="MRE"
VILT_NAME="dandelin/vilt-b32-mlm"

CUDA_VISIBLE_DEVICES=0 python -u run.py \
        --dataset_name=${DATASET_NAME} \
        --model_name=${VILT_NAME} \
        --num_epochs=15 \
        --batch_size=64 \
        --lr=0.001 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=1234 \
        --do_train \
        --max_seq=80 \
        --use_prompt \
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --save_path='ckpt/re/'