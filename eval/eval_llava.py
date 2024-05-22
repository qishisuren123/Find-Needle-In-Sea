import sys
sys.path.append('3rdparty/LLaVA')

import os
import json
import time
import argparse
import subprocess
import torch

from PIL import Image
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from rag import rag
from tools import get_input

# 设置 max_split_size_mb
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

CONV_TEMPLATE = {
    'llava-v1.5-13b': 'vicuna_v1',
    'llava-v1.6-vicuna-13b': 'vicuna_v1',
    'llava-v1.6-34b': 'chatml_direct',
}

NUM_HIDDEN_LAYERS = {
    'llava-v1.5-13b': 40,
    'llava-v1.6-vicuna-13b': 40,
    'llava-v1.6-34b': 60,
}

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def main(args):
    num_gpus = torch.cuda.device_count()
    args.rank = int(os.getenv('SLURM_PROCID', '0'))
    args.local_rank = args.rank % (num_gpus // args.num_gpus_per_rank)
    args.world_size = int(os.getenv('SLURM_NTASKS', '1'))
    args.local_world_size = num_gpus // args.num_gpus_per_rank

    os.environ['RANK'] = str(args.rank)
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    os.environ['LOCAL_WORLD_SIZE'] = str(args.local_world_size)

    if 'MASTER_ADDR' not in os.environ:
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        os.environ['MASTER_ADDR'] = addr
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '22110'

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=args.rank,
        world_size=args.world_size,
    )
    torch.cuda.set_device(args.local_rank)

    mode = os.path.basename(args.sample_file)
    model_name = os.path.basename(args.model_path)

    num_layers = NUM_HIDDEN_LAYERS[model_name]
    num_layers_per_gpu = num_layers // num_gpus

    visible_devices = [i for i in range(args.local_rank, num_gpus, args.local_world_size)]
    device_map = {
        f'model.layers.{i}': visible_devices[min(i // num_layers_per_gpu, len(visible_devices) - 1)]
        for i in range(num_layers)
    }
    device_map['model.vision_tower'] = visible_devices[0]
    device_map['vision_tower'] = visible_devices[0]
    device_map['vision_model'] = visible_devices[0]
    device_map['model.mm_projector'] = visible_devices[0]
    device_map['model.norm'] = visible_devices[0]
    device_map['model.image_newline'] = visible_devices[0]
    device_map['model.embed_tokens'] = visible_devices[0]
    device_map['lm_head'] = visible_devices[-1]

    if args.rank == 0:
        for k, v in device_map.items():
            print(k, v)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path),
        # device=args.local_rank,
        device_map=device_map,
    )
    tokenizer.model_max_length = 256000
    model.config.tokenizer_model_max_length = 256000

    print(
        f"Rank [{args.rank}] "
        f"Begin to eval model {args.model_path} on task {mode.split('.')[0]}, "
        f"using cuda:{args.local_rank}, conv_template: {CONV_TEMPLATE[model_name]}, "
        f"devices: {set([p.device for p in model.parameters()])}"
    )

    ans_name = model_name + '_' + mode
    if args.rag == "True":
        ans_name = model_name + 'ragged_' + mode
    ans_file = os.path.join(args.ans_file, ans_name)
    temp_dir = f"temp_{model_name}_{mode.replace('.jsonl', '')}"
    local_ans_file = os.path.join(args.ans_file, temp_dir, f"{args.rank}_{args.world_size}_{ans_name}")
    print('model:', model_name)
    print('mode', mode)
    print('context_len', context_len)

    os.makedirs(args.ans_file, exist_ok=True)
    os.makedirs(os.path.join(args.ans_file, temp_dir), exist_ok=True)
    with open(args.sample_file, 'r') as file:
        lines = file.readlines()

    local_file = open(local_ans_file, 'w')

    outputs_list = []
    for data in tqdm(lines[args.rank::args.world_size], desc=f"Processing {ans_name}", disable=args.rank!=0):
        sample = json.loads(data)
        sample['context'], sample['images_list'], sample['question'], sample['answer'] = get_input(sample)
        sample['context'] = sample['context'].replace('</s>', '')
        question = sample['question']

        images_list = []
        for img in sample['images_list']:
            images_list.append(os.path.join(args.image_file, img))

        if args.rag == 'True':
            sample['context'], images_list = rag(sample['context'], images_list, question, 3000)
            print('ragging')
            print('len(image):', len(images_list))
            word = sample['context'].split()
            print('len(context):', len(word))

        # 加载图像
        images = load_images(images_list)
        image_sizes = [image.size for image in images]
        images_tensor = process_images(
            images,
            image_processor,
            model.config
        )
        if isinstance(images_tensor, torch.Tensor):
            images_tensor = images_tensor.to(model.device, dtype=torch.float16)
        else:
            images_tensor = [t.to(model.device, dtype=torch.float16) for t in images_tensor]
        
        # 获取对话内容
        qs = sample['context'] + '\n' + question
        
        conv_mode = CONV_TEMPLATE[model_name]
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Token化输入
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(args.local_rank)

        try:
            output_ids = model.generate(
                inputs=input_ids, 
                images=images_tensor, 
                image_sizes=image_sizes, 
                do_sample=False,
                use_cache=True,
                max_new_tokens=32,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        except torch.cuda.OutOfMemoryError:
            print(f"Rank {args.rank} OutOfMemoryError occurs! {input_ids.shape=}, totoal_tokens={sample['meta']['context_length']}")
            outputs = 'None'

        outputs = outputs.strip()
        print(f"{input_ids.shape=}, totoal_tokens={sample['meta']['context_length']}, {outputs=}")
        outputs_list.append(json.dumps({
            "question_id": sample['id'],
            "question": question,
            "answer": sample['answer'],
            "response": outputs,
            'total_tokens':sample['meta']['context_length'],
            'position':sample['meta']['placed_depth']
        }) + "\n")
        local_file.write(outputs_list[-1])
        local_file.flush()

    print(f"Rank {args.rank} Finish")
    local_file.close()

    time.sleep(60)
    torch.distributed.barrier()

    # if args.world_size > 1:
    #     merged_outputs = [None for _ in range(args.world_size)]
    #     torch.distributed.all_gather_object(merged_outputs, outputs_list)

    #     merged_outputs = sum(merged_outputs, start=[])

    if args.rank == 0:
        print(f"Rank {args.rank} begin to merge outputs...")
        os.system(f"cat {os.path.join(args.ans_file, temp_dir)}/* > {ans_file}")
        # with open(ans_file, 'w') as file:
        #     for outputs in merged_outputs:
        #         file.write(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument('--model_path', type=str, default='/mnt/petrelfs/renyiming/model/LLaVA1/llava_vila13b')
    parser.add_argument('--image_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/img_mul_needle/')
    parser.add_argument('--sample_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/llavastyle/it.jsonl')
    parser.add_argument('--ans_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answer/llava-it.json')
    parser.add_argument('--rag', type=str, default='False')
    parser.add_argument('--num-gpus-per-rank', type=int, default=2)
    args = parser.parse_args()
    main(args)
