import sys
sys.path.append('3rdparty/LLaVA')

import os
import json
import argparse
import subprocess
import torch
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchvision.transforms.functional import InterpolationMode

from rag import rag
from tools import get_input

# 设置 max_split_size_mb
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def build_model(args):
    num_gpus = torch.cuda.device_count()
    visible_devices = [i for i in range(args.local_rank, num_gpus, args.local_world_size)]

    if len(visible_devices) > 1:
        device_map = {}
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

        num_layers = config.llm_config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu + len(visible_devices) // 2, len(visible_devices) - 1)
            device_map[f'language_model.model.layers.{i}'] = visible_devices[device_idx]

        num_layers = config.vision_config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu, len(visible_devices) // 2 - 1)
            device_map[f'vision_model.encoder.layers.{i}'] = visible_devices[device_idx]

        device_map['vision_model.embeddings'] = 0
        device_map['mlp1'] = len(visible_devices) // 2
        device_map['language_model.model.tok_embeddings'] = len(visible_devices) // 2
        device_map['language_model.model.norm'] = visible_devices[-1]
        device_map['language_model.output'] = visible_devices[-1]

    else:
        device_map = {'': visible_devices[0]}

    if args.rank == 0:
        for k, v in device_map.items():
            print(k, v)

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.model_max_length = 256000

    return model, tokenizer


def chat(
    model,
    tokenizer,
    pixel_values,
    num_patches_list,
    question,
    generation_config,
    history=None,
    return_history=False,
    IMG_START_TOKEN='<img>',
    IMG_END_TOKEN='</img>',
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
):
    assert len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    if tokenizer.convert_tokens_to_ids('<|im_end|>') != 0:
        eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')  # 92542, InternLM2
    else:
        eos_token_id = tokenizer.eos_token_id

    from internvl.conversation import get_conv_template

    template = get_conv_template(model.template)
    if history is None:
        history = []
        for num_patches in num_patches_list:
            assert '<image>' in question
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            question = question.replace('<image>', image_tokens, 1)
    else:
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].cuda()
    attention_mask = model_inputs['attention_mask'].cuda()
    generation_config['eos_token_id'] = eos_token_id

    print(f'dynamic ViT batch size: {pixel_values.size(0)}, input_ids: {input_ids.shape}')
    generation_output = model.generate(
        pixel_values=pixel_values.to(torch.bfloat16),
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    response = response.split('<|im_end|>')[0].strip()  # for InternLM2
    history.append((question, response))
    if return_history:
        return response, history
    return response


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
    model, tokenizer = build_model(args)

    print(
        f"Rank [{args.rank}] "
        f"Begin to eval model {args.model_path} on task {mode.split('.')[0]}, "
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

        if args.rag == 'True':
            images_list = []
            for img in sample['images_list']:
                images_list.append(os.path.join(args.image_file, img))
            sample['context'], images_list = rag(sample['context'], images_list, question, 3000)
            print('ragging')
            print('len(image):', len(images_list))
            word = sample['context'].split()
            print('len(context):', len(word))

        # 加载图像
        pixel_values = []
        num_patches_list = []
        for img in sample['images_list']:
            curr_pixel_values = load_image(os.path.join(args.image_file, img))
            pixel_values.append(curr_pixel_values)
            num_patches_list.append(len(curr_pixel_values))
        pixel_values = torch.cat(pixel_values)

        # 获取对话内容
        qs = sample['context'] + '\n' + question

        try:
            outputs = chat(
                model=model,
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=qs,
                generation_config=dict(
                    do_sample=False,
                    max_new_tokens=32,
                ),
            )
        except torch.cuda.OutOfMemoryError:
            print(f"Rank {args.rank} OutOfMemoryError occurs! totoal_tokens={sample['meta']['context_length']}")
            outputs = 'None'

        outputs = outputs.strip()
        print(f"totoal_tokens={sample['meta']['context_length']}, {outputs=}")
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
