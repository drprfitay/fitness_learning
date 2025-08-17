#!/bin/bash
source ~/.bashrc
conda activate esm_env
python ../code/train_epinnet.py \
    --train False \
    --evaluate_train True \
    --evaluate_test True \
    --save_path "pretraining/esm8m/zero_shot/train_12_test_345" \
    --train_indices "[1,2]" \
    --test_indices "[3,4,5]" 
