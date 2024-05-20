#!/bin/bash

# 常量路径
# image_file='/mnt/petrelfs/renyiming/LTT/NeedleInSea/data/img_mul_needle/'
# image_file='/mnt/petrelfs/share_data/duanyuchen/datasets/mm_niah/'
# image_file='/mnt/petrelfs/share_data/wangweiyun/share_projects/mm_niah/save_long_aug_vqa_05_01_img_local_paths/'
image_file='/mnt/petrelfs/share_data/renyiming/'

ans_file='/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answerv3'

# 循环不同的数据集和答案文件
declare -a model_paths=('/mnt/petrelfs/renyiming/model/LLaVA1/llava_1.5' \
                        '/mnt/petrelfs/renyiming/model/LLaVA16/llava_1.6' \
                        '/mnt/petrelfs/renyiming/model/LLaVA1/llava_vila13b')

declare -a sample_files=('/mnt/petrelfs/share_data/renyiming/llavastyle/ct.jsonl' \
                        '/mnt/petrelfs/share_data/renyiming/llavastyle/ci.jsonl' )


# # 确保 sample_files 和 ans_files 数组长度相等
# for ((i=0; i<${#model_paths[@]}; i++)); do
#     for ((j=0; j<${#sample_files[@]}; j++)); do
#         model_path=${model_paths[i]}
#         sample_file=${sample_files[j]}

#         srun -p INTERN3 \
#             --gres=gpu:1 \
#             --quotatype=spot \
#             python /mnt/petrelfs/renyiming/dataset/sea-needle/eval/eval_llava.py \
#             --model_path $model_path \
#             --image_file $image_file \
#             --sample_file $sample_file \
#             --ans_file $ans_file \
#             --rag True
#     done
# done

for ((i=0; i<${#model_paths[@]}; i++)); do
    for ((j=0; j<${#sample_files[@]}; j++)); do
        model_path=${model_paths[i]}
        sample_file=${sample_files[j]}

        srun -p INTERN3 \
            --gres=gpu:1 \
            --quotatype=spot \
            python /mnt/petrelfs/renyiming/dataset/sea-needle/eval/eval_llava.py \
            --model_path $model_path \
            --image_file $image_file \
            --sample_file $sample_file \
            --ans_file $ans_file \
            --rag False
    done
done


