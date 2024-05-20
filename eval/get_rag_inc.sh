#!/bin/bash

# 常量路径

image_file='/mnt/petrelfs/share_data/zhangshuibai/05_13_img_local_paths/'
ans_file='/mnt/petrelfs/renyiming/LTT/NeedleInSea/llava_style_new/llavastyle'

# 循环不同的数据集和答案文件
declare -a model_paths=('/mnt/petrelfs/renyiming/model/LLaVA1/llava_1.5')

declare -a sample_files=('/mnt/petrelfs/share_data/zhangshuibai/0513_merged_vqa_aug_file.jsonl')

# declare -a sample_files=('/mnt/petrelfs/share_data/wangweiyun/share_projects/mm_niah/merged_vqa_aug_file.jsonl')
# 确保 sample_files 和 ans_files 数组长度相等
for ((i=0; i<${#model_paths[@]}; i++)); do
    for ((j=0; j<${#sample_files[@]}; j++)); do
        model_path=${model_paths[i]}
        sample_file=${sample_files[j]}

        srun -p INTERN3 \
            --gres=gpu:1 \
            --quotatype=spot \
            python /mnt/petrelfs/renyiming/dataset/sea-needle/eval/getRAG.py \
            --model_path $model_path \
            --image_file $image_file \
            --sample_file $sample_file \
            --ans_file $ans_file
    done
done
