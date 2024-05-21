#!/bin/bash

PARTITION=${PARTITION:-"llm_s"}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-2}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

# 常量路径
image_file='data'
ans_file="outputs_${GPUS}"

# 循环不同的数据集和答案文件
declare -a model_paths=( \
    # 'ckpts/liuhaotian/llava-v1.5-13b' \
    'ckpts/liuhaotian/llava-v1.6-vicuna-13b' \
    # 'ckpts/liuhaotian/llava-v1.6-34b' \
    # 'ckpts/Efficient-Large-Model/VILA1.5-13b' \
    # 'ckpts/Efficient-Large-Model/VILA1.5-40b' \
)

declare -a sample_files=( \
    'data/annotations/it.jsonl' \
    # 'data/annotations/ii.jsonl' \
    # 'data/annotations/ct.jsonl' \
    # 'data/annotations/ci.jsonl' \
    # 'data/annotations/infer-choose.jsonl' \
    # 'data/annotations/visual-reasoning.jsonl' \
    # 'data/annotations/ragged_it.jsonl' \
    # 'data/annotations/ragged_ii.jsonl' \
    # 'data/annotations/ragged_ct.jsonl' \
    # 'data/annotations/ragged_ci.jsonl' \
    # 'data/annotations/ragged__infer-choose.jsonl' \
    # 'data/annotations/ragged__visual-reasoning.jsonl' \
)

mkdir -p logs

# 确保 sample_files 和 ans_files 数组长度相等
for ((i=0; i<${#model_paths[@]}; i++)); do
    for ((j=0; j<${#sample_files[@]}; j++)); do
        model_path=${model_paths[i]}
        sample_file=${sample_files[j]}

        srun -p ${PARTITION} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=$((GPUS / GPUS_PER_TASK)) \
            --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
            --quotatype=${QUOTA_TYPE} \
            --job-name="eval_$(basename ${sample_file} .${sample_file##*.})" \
            python -u eval/eval_llava.py \
            --model_path $model_path \
            --image_file $image_file \
            --sample_file $sample_file \
            --ans_file $ans_file \
            --rag False \
            --num-gpus-per-rank ${GPUS_PER_TASK} \
            2>&1 | tee -a "logs/$(basename ${model_path})_$(basename ${sample_file} .${sample_file##*.})_wo_rag_${GPUS}.log"
    done
done

# for ((i=0; i<${#model_paths[@]}; i++)); do
#     for ((j=0; j<${#sample_files[@]}; j++)); do
#         model_path=${model_paths[i]}
#         sample_file=${sample_files[j]}

#         srun -p ${PARTITION} \
#             --gres=gpu:${GPUS_PER_NODE} \
#             --ntasks=${GPUS} \
#             --ntasks-per-node=${GPUS_PER_NODE} \
#             --quotatype=${QUOTA_TYPE} \
#             --job-name="eval_$(basename ${sample_file} .${sample_file##*.})" \
#             --async \
#             python -u eval/eval_llava.py \
#             --model_path $model_path \
#             --image_file $image_file \
#             --sample_file $sample_file \
#             --ans_file $ans_file \
#             --rag True \
#             2>&1 | tee -a "logs/$(basename ${model_path})_$(basename ${sample_file} .${sample_file##*.})_with_rag_${GPUS}.log"
#     done
# done
