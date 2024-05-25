#!/bin/bash

PARTITION=${PARTITION:-"llm_s"}
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-4}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

# 常量路径
image_file='data'
ans_file="outputs_ragged_${GPUS}"
split_dir="outputs_text_only_rag_64"

# 循环不同的数据集和答案文件
declare -a model_paths=( \
    'ckpts/liuhaotian/llava-v1.5-13b' \
    # 'ckpts/liuhaotian/llava-v1.6-vicuna-13b' \
    # 'ckpts/Efficient-Large-Model/VILA1.0-13b-llava' \
    # 'ckpts/Efficient-Large-Model/VILA1.5-13b' \
    # 'ckpts/liuhaotian/llava-v1.6-34b' \
    # 'ckpts/Efficient-Large-Model/VILA1.5-40b' \
)

declare -a split_files=( \
    # 'it_debug.jsonl' \
    # 'it.jsonl' \
    # 'ii.jsonl' \
    # 'ct.jsonl' \
    # 'ci.jsonl' \
    'infer-choose.jsonl' \
    'visual-reasoning.jsonl' \
)

mkdir -p logs_ragged_${GPUS}

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
            python -u eval/eval_llava.py \
            --model_path $model_path \
            --image_file $image_file \
            --sample_file "${split_dir}/${model_name}_ragged_${split_file}" \
            --ans_file $ans_file \
            --rag False \
            --num-gpus-per-rank ${GPUS_PER_TASK} \
            2>&1 | tee -a "logs_ragged_${GPUS}/${model_name}_${split_name}_wo_rag.log"

        cat ${ans_file}/temp_${model_name}_${model_name}_ragged_${split_name}/* > ${ans_file}/${model_name}_ragged_${split_name}.jsonl
    done
done