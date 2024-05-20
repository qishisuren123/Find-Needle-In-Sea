#!/bin/bash

# 常量路径
# image_file='/mnt/petrelfs/renyiming/LTT/NeedleInSea/data/img_mul_needle/'
image_file='/mnt/petrelfs/share_data/duanyuchen/datasets/mm_niah/'
# image_file='/mnt/petrelfs/share_data/wangweiyun/share_projects/mm_niah/save_long_aug_vqa_05_01_img_local_paths/'
# image_file='/mnt/petrelfs/share_data/renyiming/'

ans_file='/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answer'

# 确保 sample_files 和 ans_files 数组长度相等

srun -p INTERN3 \
    --gres=gpu:2 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    python /mnt/petrelfs/renyiming/dataset/sea-needle/eval/eval_emu2.py \
    --image_file $image_file \
    --ans_file $ans_file \
    --rag True

srun -p INTERN3 \
    --gres=gpu:2 \
    python /mnt/petrelfs/renyiming/dataset/sea-needle/eval/eval_emu2.py \
    --image_file $image_file \
    --ans_file $ans_file \
    --rag False
