import io
import os
import json
import base64
import torch
import tqdm

from torchvision import transforms
from transformers import AutoTokenizer
from dataset import InterleavedDataset, TCSLoader
from dataset import (
    BOX_END_TOKEN, BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
    IMG_START_TOKEN, QUAD_END_TOKEN,
    QUAD_START_TOKEN, REF_END_TOKEN,
    REF_START_TOKEN
)
from torch.utils.data import DataLoader
import os
import os.path as osp
import yaml
import argparse
import torch
import pandas as pd
from uuid import uuid4
from openai import OpenAI
from tqdm import tqdm
import pdb

import random
import time
import re
from PIL import Image
import base64
import tiktoken

os.environ["OPENAI_API_KEY"] = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_BASE_URL"] = "https://api.openai-sb.com/v1"


_filepath = '/mnt/petrelfs/liushuo/VQAG/obelisc_10m.json'

MEAN = (0.485, 0.456, 0.406)
MEAN = torch.tensor(MEAN).view(3, 1, 1)
STD = (0.229, 0.224, 0.225)
STD = torch.tensor(STD).view(3, 1, 1)

TEMPLATE_NAME = 'plain_internlm2'
MODEL = 'internlm/internlm2-chat-7b'
IMAGE_SIZE = 224
PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 1
MAX_NUM_IMAGES = 100

NUM_IMG_TOKEN = int((IMAGE_SIZE // PATCH_SIZE * DOWNSAMPLE_RATIO) ** 2)

tcs_loader = TCSLoader('~/petreloss.conf')
unloader = transforms.ToPILImage()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)

token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
              QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
              REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
tokenizer.add_tokens(token_list, special_tokens=True)



def encode_image_file_to_base64(image_path):
    if image_path.endswith('.png'):
        tmp_name = f'{timestr(second=True)}.jpg'
        img = Image.open(image_path)
        img.save(tmp_name)
        result = encode_image_file_to_base64(tmp_name)
        os.remove(tmp_name)
        return result
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        
    encoded_image = base64.b64encode(image_data)
    return encoded_image.decode('utf-8')

def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size to (target_size, target_size)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        #img = img.resize((target_size, target_size))
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    ret = encode_image_file_to_base64(tmp)
    os.remove(tmp)
    return ret
    


def call_gpt(history_chat, model="gpt-4", temp_gpt=0.0):
    client = OpenAI()
    chatgpt_messages = []
    for role, message in history_chat:
        ret = dict()
        if message:
            ret["role"] = role
            ret['content'] = message
            chatgpt_messages.append(ret)    
    success = False
    while not success:
        try:
            response = client.chat.completions.create(model="gpt-4-vision-preview",messages = chatgpt_messages,max_tokens=512)
            #print("chatgpt_messages",chatgpt_messages)
            reply = response.choices[0].message.content
            print("reply", reply)
            total_tokens = response.usage.total_tokens
            success = True
            return reply, total_tokens
        except Exception as e:
            print('[Worker] an exception occured: %s (%s). retrying in 3 minutes.' % (type(e), str(e)))
            time.sleep(30)
def parse_final_answer(gpt_response):
    # ===== Parse the paragraph starting with analysis. =====
    analysis_answer_result = re.search('Answer:(.*)', gpt_response)
    answer = analysis_answer_result.group(1).strip()       
    return answer 
    '''
    try:
        analysis_question_result = re.search('Question:(.*)\n', gpt_response)
        print("analysis_question_result",analysis_question_result)
        question = analysis_question_result.group(1).strip()
        print("question",question)
        analysis_answer_result = re.search('Answer:(.*)\n', gpt_response)
        print("analysis_answer_result",analysis_answer_result)
        answer = analysis_answer_result.group(1).strip()      
        print("answer",answer)  
        return question, answer
    except Exception as e:
        print("Can not parse analysis")
        return None
    '''

def LongDC(dataset, data_id, token_max_list, long_save_path):
    long_document_image = []
    long_docment_text = []
    long_document_valid_image = []
    long_document_metadata = []
    token_max_len = len(token_max_list)
    for token_max in token_max_list:
        long_save_path_token_max = os.path.join(long_save_path, str(token_max))
        if not os.path.exists(long_save_path_token_max):
            os.makedirs(long_save_path_token_max)
        print("long_save_path_token_max", long_save_path_token_max)
        long_document_path = os.path.join(long_save_path,  str(token_max), '{}.yaml'.format(data_id))    
        print("long_document_path", long_document_path)
        if os.path.isfile(long_document_path):
            break


        ids = []
        current_id = data_id
        token_num = 0
        flag = 0
        while current_id<100:
            ids.append(current_id)
            ret = dataset.fetch_data(current_id) 
            images = ret['images_ori']
            texts = ret['texts']    
            valid_image = ret['valid_image'] 
            metadata = ret['metadata']

            image_idx = 0
            valid_image_idx = 0
            print(len(texts))
            print(len(images))
            print(len(valid_image))
            for i in range(len(texts)):
                if texts[i] is None:
                    long_docment_text.append(texts[i])    
                    print("image_idx", image_idx)
                    long_document_valid_image.append(valid_image[valid_image_idx])
                    long_document_metadata.append(metadata[valid_image_idx])
                    if valid_image[valid_image_idx]:
                        long_document_image.append(images[image_idx])
                        token_num+=85
                        image_idx += 1  
                    valid_image_idx += 1   
                else:
                    long_docment_text.append(texts[i])    
                    enc = tiktoken.encoding_for_model("gpt-4")   
                    tokens = enc.encode(texts[i])
                    token_num += len(tokens)
                if token_num>token_max:
                    flag = 1
                    break
            print(len(long_docment_text))
            print(len(long_document_image))
            print(len(long_document_valid_image))
            if flag == 1:
                break
            current_id+=1

        results = {}
        results["token_num"] = token_num
        results["ids"] = ids
        results['texts'] = long_docment_text
        results['valid_image'] = long_document_valid_image
        results['images_ori'] = long_document_image          
        results['metadata'] = long_document_metadata  

        result_path = os.path.join(long_save_path, str(token_max), '{}.yaml'.format(data_id))
        with open(result_path, 'w') as f:
                yaml.dump(results, f)


def Eval(dataset, data_ids, token_max, vqa_path  ='', save_path='', long_save_path=''):
    for data_id in data_ids:
        LongDC(dataset, data_id, token_max, long_save_path)   



if __name__ == "__main__":
    with open(_filepath) as file:
        _filepath = json.load(file)
    print(_filepath["obelisc_10m"]['data_augment'])
    ann_filename = "obelisc_10m"
    dataset = InterleavedDataset(
        template_name=TEMPLATE_NAME,
        meta=_filepath[ann_filename],
        tokenizer=tokenizer,
        tcs_loader=tcs_loader,
        num_image_token=NUM_IMG_TOKEN,
        image_size=IMAGE_SIZE,
        is_train=_filepath[ann_filename]['data_augment'],
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        max_num_images=MAX_NUM_IMAGES,
    )

    data_ids = list(dataset.ann.keys())
    print('data_ids', data_ids)
    save_path = "/mnt/lustre/liushuo/VQAG/result"
    long_save_path = "/mnt/lustre/liushuo/VQAG/long_document"
    vqa_path = "/mnt/lustre/liushuo/VQAG/vqa"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(long_save_path):
        os.makedirs(long_save_path)
    # start Conversation
    token_max_list = [1000, 2000, 3000, 5000, 9000]
    Eval(dataset, data_ids, token_max_list, vqa_path= vqa_path, save_path=save_path, long_save_path=long_save_path)    

