#!/bin/bash

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-4}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

# 常量路径
image_file='mm-niah-v1/images'
ans_file="outputs_v1_${GPUS}"

# 循环不同的数据集和答案文件
declare -a model_paths=( \
    'ckpts/OpenGVLab/InternVL-Chat-V1-5' \
)

declare -a split_files=( \
    # 'data/annotations/it_debug.jsonl' \
    # 'data/annotations/it.jsonl' \
    # 'data/annotations/ii.jsonl' \
    # 'data/annotations/ct.jsonl' \
    # 'data/annotations/ci.jsonl' \
    # 'mm-niah-v1/annotations/find-text.jsonl' \
    'mm-niah-v1/annotations/find-image.jsonl' \
    'mm-niah-v1/annotations/count-text-easy.jsonl' \
    'mm-niah-v1/annotations/count-image-easy.jsonl' \
    # 'data/annotations/infer-choose.jsonl' \
    # 'data/annotations/visual-reasoning.jsonl' \
    # 'data/annotations/ragged_it.jsonl' \
    # 'data/annotations/ragged_ii.jsonl' \
    # 'data/annotations/ragged_ct.jsonl' \
    # 'data/annotations/ragged_ci.jsonl' \
    # 'data/annotations/ragged__infer-choose.jsonl' \
    # 'data/annotations/ragged__visual-reasoning.jsonl' \
)

mkdir -p logs_v1_${GPUS}

# 确保 split_files 和 ans_files 数组长度相等
for ((i=0; i<${#model_paths[@]}; i++)); do
    for ((j=0; j<${#split_files[@]}; j++)); do
        model_path=${model_paths[i]}
        split_file=${split_files[j]}

        model_name="$(basename ${model_path})"
        split_name="$(basename ${split_file} .${split_file##*.})"

        srun -p ${PARTITION} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=$((GPUS / GPUS_PER_TASK)) \
            --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
            --quotatype=${QUOTA_TYPE} \
            --job-name="eval_${split_name}" \
            python -u eval/eval_internvl.py \
            --model_path $model_path \
            --image_file $image_file \
            --sample_file $split_file \
            --ans_file $ans_file \
            --rag False \
            --num-gpus-per-rank ${GPUS_PER_TASK} \
            2>&1 | tee -a "logs_v1_${GPUS}/${model_name}_${split_name}_wo_rag.log"

        cat ${ans_file}/temp_${model_name}_${split_name}/* > ${ans_file}/${model_name}_${split_name}.jsonl
    done
done
