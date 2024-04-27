import io
import os
import json
import gradio as gr
import base64
import torch

from torchvision import transforms
from transformers import AutoTokenizer
from dataset_vis_needle import InterleavedDataset
from dataset_vis_needle import (
    BOX_END_TOKEN, BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
    IMG_START_TOKEN, QUAD_END_TOKEN,
    QUAD_START_TOKEN, REF_END_TOKEN,
    REF_START_TOKEN
)

filepath = '/mnt/petrelfs/renyiming/dataset/sea-needle/data/obelisc_10m.json'

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

unloader = transforms.ToPILImage()
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)

token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
              QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
              REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
tokenizer.add_tokens(token_list, special_tokens=True)

def unload_pixel_value(pixel_value):
    image = unloader(pixel_value * STD + MEAN)
    return image

def image_to_mdstring(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"![image](data:image/jpeg;base64,{img_str})"

def process_item(item, image_placeholder):
    text = tokenizer.decode(item['input_ids'])
    label = tokenizer.decode(torch.where(item['labels'] < 0, tokenizer.unk_token_id, item['labels']))
    all_images = [unload_pixel_value(image) for image in item['pixel_values']]

    num_image_placeholders = text.count(image_placeholder)
    assert num_image_placeholders == len(all_images), (text, num_image_placeholders, len(all_images))

    for i in range(num_image_placeholders):
        text = text.replace(image_placeholder, image_to_mdstring(all_images[i]), 1)

    md_str = [
        '## Meta Info',
        f"{item['input_ids'].shape=}",
        f"{item['labels'].shape=}",
        f"num_image_tokens={num_image_placeholders * NUM_IMG_TOKEN}",
        f"num_text_tokens={item['input_ids'].shape[0] - num_image_placeholders * NUM_IMG_TOKEN}",
        '## Input', text,
        '## Target', label,
    ]
    md_str = '\n\n'.join(md_str)

    return md_str.replace('<', '\\<').replace('>', '\\>')



def gradio_app_vis_incontext_trainset():
    data = InterleavedDataset(
        template_name=TEMPLATE_NAME,
        tokenizer=tokenizer,
        num_image_token=NUM_IMG_TOKEN,
        image_size=IMAGE_SIZE,
        is_train=False,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        max_num_images=MAX_NUM_IMAGES,
    )
    
    def when_btn_submit_click(ann_id, md_annotation):
        try:
            item = data[int(max(min(ann_id, len(data) - 1), 0))]
        except IndexError as err:
            print(ann_id, len(data), int(max(min(ann_id, len(data) - 1), 0)))
            raise err
        md_annotation = process_item(item, '> <'.join(data.image_tokens.split('><')))
        return int(max(min(ann_id, len(data) - 1), 0)), md_annotation

    def when_btn_next_click(ann_id, md_annotation):
        return when_btn_submit_click(ann_id + 1, md_annotation)
        
    with gr.Blocks() as app:
       
        with gr.Row():
            ann_id = gr.Number(0)
            btn_next = gr.Button("Next")
            btn_submit = gr.Button("id跳转")
        annotation = gr.Markdown()

        all_components = [ann_id, annotation]
        btn_submit.click(when_btn_submit_click, all_components, all_components)
        btn_next.click(when_btn_next_click, all_components, all_components)

    server_port = 10015
    for i in range(10015, 10100):
        cmd = f'netstat -aon|grep {i}'
        with os.popen(cmd, 'r') as file:
            if '' == file.read():
                server_port = i
                break
    app.launch(share=True, server_port=server_port)
    


if __name__ == "__main__":
    print('begin')
    gradio_app_vis_incontext_trainset()
