import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json
from tqdm import tqdm
from string import Template
from PIL import Image
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
import numpy as np
import re
import shortuuid
import argparse
import os

from tools import get_input
from rag import rag

# 设置 max_split_size_mb
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def replace_first_image(string, replacement):
    # 查找第一个 <image> 的位置
    index = string.find("<image>")
    if index != -1:  # 如果找到了 <image>，则替换
        # 将 <image> 及其后的部分替换为指定的字符串
        string = string[:index] + replacement + string[index + len("<image>"):]
    return string

def get_qs(sample):
    if sample['meta']['choices'] == None:
        return sample['question']
    else:
        qs = sample['question'] + "(Please choose an object from:"
        for it in sample['meta']['choices']:
            qs = qs + it + ', '
        qs += ")"
        return qs


# model_path = "/mnt/petrelfs/renyiming/model/LLaVA/llava_model"
model_path = '/mnt/petrelfs/renyiming/model/LLaVA1/llava_vila13b'
image_file = '/mnt/petrelfs/renyiming/dataset/sea-needle/img_mul_needle/'
sample_file = '/mnt/petrelfs/renyiming/dataset/sea-needle/llavastyle/it.jsonl'
ans_file = '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answer/llava-it.json'



def main(args):

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path),
        device="cuda"
    )
    model_name = os.path.basename(args.model_path)
    mode = os.path.basename(args.sample_file)
    ans_name = model_name + '_' + mode
    if args.rag=="True":
        ans_name = model_name + 'ragged_' + mode
    ans_file = os.path.join(args.ans_file, ans_name)
    print('model:', model_name)
    print('mode', mode)
        

    with open(args.sample_file, 'r') as file, open(ans_file, 'w') as ans_file:
        for data in tqdm(file, desc="Processing "+ans_name):
            sample = json.loads(data)
            sample['context'], sample['images_list'], sample['question'], sample['answer'] = get_input(sample)
            question = sample['question']
            
            images_list = []
            for img in sample['images_list']:
                images_list.append(args.image_file + img)
            try:
                if args.rag=='True':
                    sample['context'], images_list = rag(sample['context'], images_list, question, 3000)
                    print('ragging')
                    print('len(image):', len(images_list))
                    word = sample['context'].split()
                    print('len(context):', len(word))
                # 加载图像
                images = load_images(images_list)
                images_tensor = process_images(
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
                
                # 获取对话内容
                qs = sample['context'] + question
                
                conv_mode = "vicuna_v1"
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                # Token化输入
                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )

                # 无梯度生成输出
                with torch.no_grad():
                    output_ids = model.generate(input_ids=input_ids, 
                                                images=images_tensor, 
                                                labels=input_ids,
                                                max_new_tokens=200)
                    
                # 检查输出与输入的差异
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                
                # 解码输出
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
            except Exception as e:
                print(e)
                outputs = "None"

            print(outputs)
            ans_file.write(json.dumps({"question_id": sample['id'],
                                        "answer": sample['answer'],
                                        "response": outputs,
                                        'total_tokens':sample['meta']['context_length'],
                                        'position':sample['meta']['placed_depth']
                                        }) + "\n")
            
def main_rag(args):

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path),
        device="cuda"
    )
    model_name = os.path.basename(args.model_path)
    mode = os.path.basename(args.sample_file)
    ans_name = model_name + '_' + mode
    
    ans_file = os.path.join(args.ans_file, ans_name)
    print('model:', model_name)
    print('mode', mode)
        

    with open(args.sample_file, 'r') as file, open(ans_file, 'w') as ans_file:
        for data in tqdm(file, desc="Processing "+ans_name):
            sample = json.loads(data)
            question = get_qs(sample)
            
            if args.rag=="True":
                sample['context'], sample['images_list'] = rag(sample['context'], question)
            
            images_list = []
            for img in sample['images_list']:
                index = img.find('img_mul_needle/')
                images_list.append(args.image_file + img[index+len('img_mul_needle/'):])
            try:
                # 加载图像
                images = load_images(images_list)
                images_tensor = process_images(
                    images,
                    image_processor,
                    model.config
                ).to(model.device, dtype=torch.float16)
                
                # 获取对话内容
                qs = sample['context'] + "Question:" + question
                
                conv_mode = "vicuna_v1"
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                
                # Token化输入
                input_ids = (
                    tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                    .unsqueeze(0)
                    .cuda()
                )

                # 无梯度生成输出
                with torch.no_grad():
                    output_ids = model.generate(input_ids=input_ids, 
                                                images=images_tensor, 
                                                labels=input_ids,
                                                max_new_tokens=96)
                    
                # 检查输出与输入的差异
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                
                # 解码输出
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
            except Exception as e:
                print(e)
                outputs = "None"

            print(outputs)
            ans_file.write(json.dumps({"question_id": sample['id'],
                                        "answer": sample['answer'],
                                        "response": outputs,
                                        'total_tokens':sample['meta']['context_length'],
                                        'position':sample['meta']['placed_depth']
                                        }) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument('--model_path', type=str, default='/mnt/petrelfs/renyiming/model/LLaVA1/llava_vila13b')
    parser.add_argument('--image_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/img_mul_needle/')
    parser.add_argument('--sample_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/llavastyle/it.jsonl')
    parser.add_argument('--ans_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answer/llava-it.json')
    parser.add_argument('--rag', type=str, default='False')
    args = parser.parse_args()
    main(args)
