
import os
import json
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

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from tools import get_input

CONV_TEMPLATE = {
    'llava-v1.5-13b': 'vicuna_v1',
    'llava-v1.6-vicuna-13b': 'vicuna_v1',
    'llava-v1.6-34b': 'chatml_direct',
    'VILA1.0-13b-llava': 'vicuna_v1',
}

NUM_HIDDEN_LAYERS = {
    'llava-v1.5-13b': 40,
    'llava-v1.6-vicuna-13b': 40,
    'llava-v1.6-34b': 60,
    'VILA1.0-13b-llava': 40,
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

def replace_first_image(string, replacement):
    assert '<image>' in string
    return string.replace('<image>', replacement, 1)
     
def main_rag(args):
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
    model.config.max_length = 256000
    model.config.tokenizer_model_max_length = 256000

    # if model.config.image_aspect_ratio == 'anyres':
    #     model.config.image_aspect_ratio = 'pad'
    #     model.config.mm_patch_merge_type = 'flat'

    print(
        f"Rank [{args.rank}] "
        f"Begin to eval model {args.model_path} on task {mode.split('.')[0]}, "
        f"using cuda:{args.local_rank}, conv_template: {CONV_TEMPLATE[model_name]}, "
        f"devices: {set([p.device for p in model.parameters()])}"
    )

    ans_name = model_name + 'rag_' + mode
    ans_file = os.path.join(args.ans_file, ans_name)
    temp_dir = f"temp_rag_{model_name}_{mode.replace('.jsonl', '')}"
    local_ans_file = os.path.join(args.ans_file, temp_dir, f"{args.rank}_{args.world_size}_{ans_name}")
    print('model:', model_name)
    print('mode', mode)
    print('context_len', context_len)

    os.makedirs(args.ans_file, exist_ok=True)
    os.makedirs(os.path.join(args.ans_file, temp_dir), exist_ok=True)
    args.local_ans_file = local_ans_file

    with open(args.sample_file, 'r') as file:
        lines = file.readlines()
    save_file = open(local_ans_file, 'w')

    for data in tqdm(lines[args.rank::args.world_size], desc=f"Processing {ans_name}", disable=args.rank!=0):
        sample = json.loads(data)
    
        images_list = []
        for img in sample['images_list']:
            images_list.append(os.path.join(args.image_file, img))

        num_images_in_question = sample['question'].count('<image>')
        if num_images_in_question > 0:
            images_list = images_list[:-num_images_in_question]
        
        for img in images_list:
            # 加载图像
            images = load_images([img])
            image_sizes = [image.size for image in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            ).to(model.device, dtype=torch.float16)

            # 获取对话内容
            qs = '''<image>\nYou are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Give a concise summary of the this image that is well optimized for retrieval.'''

            conv_mode = CONV_TEMPLATE[model_name]
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
            output_ids = model.generate(
                inputs=input_ids, 
                images=images_tensor, 
                image_sizes=image_sizes, 
                do_sample=False,
                use_cache=True,
                max_new_tokens=128,
            )

            # 解码输出
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()
            sample['context'] = replace_first_image(sample['context'], outputs)
            print(outputs)

        save_file.write(json.dumps(sample) + "\n")
        save_file.flush()

    save_file.close()

def rag(text, query, length=4096):
    documents = Document(page_content=text)
    top_k = length//100
    
    text_splitter =RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "?", "."],
        chunk_size=100,
        chunk_overlap=0,
    )
    texts = text_splitter.split_documents([documents])
    vectorstore = Chroma.from_documents(documents=texts, embedding=HuggingFaceEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    retrieved_docs = retriever.invoke(query)

    sim_doc = ''
    for i in range(min(top_k, len(retrieved_docs))):
        sim_doc += retrieved_docs[i].page_content
        
    return sim_doc

def pro_rag(args):
    mode = os.path.basename(args.sample_file)
    model_name = os.path.basename(args.model_path)
    ans_name = model_name + 'ragged_' + mode

    ans_file = os.path.join(args.ans_file, ans_name)
    temp_dir = f"temp_ragged_{model_name}_{mode.replace('.jsonl', '')}"
    local_ans_file = os.path.join(args.ans_file, temp_dir, f"{args.rank}_{args.world_size}_{ans_name}")

    os.makedirs(args.ans_file, exist_ok=True)
    os.makedirs(os.path.join(args.ans_file, temp_dir), exist_ok=True)

    print(f'Rank {args.rank} begin to process rag')
    with open(args.local_ans_file, 'r') as file, open(local_ans_file, 'w') as save_file:
        for data in tqdm(file, desc=f"Processing {ans_name}", disable=args.rank!=0):
            sample = json.loads(data)
            # images_list = sample['images_list']
            sample['images_list'] = []
            context, images_list, question, answer = get_input(sample)
            # sample['images_list'] = images_list
            sample['images_list'] = []
            sample['context'] = rag(sample['context'], question)

            save_file.write(json.dumps(sample) + "\n")
            save_file.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument('--model_path', type=str, default='/mnt/petrelfs/renyiming/model/LLaVA1/llava_vila13b')
    parser.add_argument('--image_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/img_mul_needle/')
    parser.add_argument('--sample_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/llavastyle/it.jsonl')
    parser.add_argument('--ans_file', type=str, default= '/mnt/petrelfs/renyiming/dataset/sea-needle/eval/answer/llava-it.json')
    parser.add_argument('--num-gpus-per-rank', type=int, default=2)
    args = parser.parse_args()
    main_rag(args)
    pro_rag(args)
