#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/ \
    --summary_dir ./result/log/ \
    --mode inference \
    --is_training False \
    --task SRGAN \
    --input_dir_LR ./myImages \
    --num_resblock 16 \
    --pre_trained_model False \
    --checkpoint #MODEL DIR NAME - NOT YET PRESENT