from PIL import Image 
import requests
import torch 
import re
from tqdm import tqdm
import json
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch


# 设置 max_split_size_mb
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def load_images(image_list):
    images = []
    for img in image_list:
        images.append(Image.open(img).convert("RGB"))
    return images

def get_qs(sample):
    if sample['meta']['choices'] == None:
        return sample['question']
    else:
        qs = sample['question'] + "(Please choose an object from:"
        for it in sample['meta']['choices']:
            qs = qs + it + ', '
        qs += ")"
        return qs
    
def main(args):
    # sample_files = args.sample_files
    # sample_files = ['/mnt/petrelfs/share_data/renyiming/llavastyle/ct.jsonl', 
    #                 '/mnt/petrelfs/share_data/renyiming/llavastyle/ci.jsonl',
    #                 '/mnt/petrelfs/share_data/renyiming/llavastyle/it.jsonl',
    #                 '/mnt/petrelfs/share_data/renyiming/llavastyle/ii.jsonl'
    #                 ]
    sample_files = ['/mnt/petrelfs/share_data/duanyuchen/mm_niah/niah/infer-choose.jsonl',
                    '/mnt/petrelfs/share_data/duanyuchen/mm_niah/niah/visual-reasoning.jsonl'
                    ]
    # sample_files = ['/mnt/petrelfs/share_data/wangweiyun/share_projects/mm_niah/merged_vqa_aug_file.jsonl']
    # sample_files = ['/mnt/petrelfs/renyiming/LTT/NeedleInSea/llava_style_new/llavastyle/ragged__infer-choose.jsonl',
    #                 '/mnt/petrelfs/renyiming/LTT/NeedleInSea/llava_style_new/llavastyle/ragged__visual-reasoning.jsonl']
    model_name = 'emu2-chat'
       
    tokenizer = AutoTokenizer.from_pretrained("/mnt/petrelfs/share_data/wangwenhai/llm/Emu2-Chat/") # "BAAI/Emu2-Chat"

    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            "/mnt/petrelfs/share_data/wangwenhai/llm/Emu2-Chat/", # "BAAI/Emu2-Chat"
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True)  

    # device_map = infer_auto_device_map(model, max_memory={0:'38GiB',1:'38GiB',2:'20GiB',3:'20GiB',4:'20GiB',5:'20GiB',6:'20GiB',7:'20GiB'}, no_split_module_classes=['Block','LlamaDecoderLayer'])  
    device_map = infer_auto_device_map(model, max_memory={0:'38GiB',1:'38GiB',}, no_split_module_classes=['Block','LlamaDecoderLayer'])  
    # input and output logits should be on same device
    device_map["model.decoder.lm.lm_head"] = 0

    model = load_checkpoint_and_dispatch(
        model, 
        '/mnt/petrelfs/share_data/wangwenhai/llm/Emu2-Chat/',
        device_map=device_map).eval()

    for sample_file in sample_files:
        mode = os.path.basename(sample_file)
        ans_name = model_name + '_' + mode
        if args.rag=="True":
            ans_name = model_name + 'ragged_' + mode
        ans_file = os.path.join(args.ans_file, ans_name)
        print('model:', model_name)
        print('mode', mode)
        
        with open(sample_file, 'r') as file, open(ans_file, 'w') as ans_file:
            for data in tqdm(file, desc="Processing "+ans_name):
                sample = json.loads(data)
                question = get_qs(sample)
                if 'ct' in mode:
                    question += 'Please help the little penquin collect the number of *, for example: {"little penquin": [x, x, ... ]). The summation is not requiredand, and the numbers in [x, x, .x...] represent the counted number of  by the little penguin. Only output the results in JSONformat without any explanation.'
            
                # sample['context'] = rag(sample['context'], question)
                
                
                #find and count without rag
                # images_list = sample['images_list']
                # images = load_images(images_list)
                
                #find and count with rag
                # sample['context'] = '[<IMG_PLH>]' + sample['context']
                # # 定义图片的大小
                # height, width = 256, 256
                # # 生成全零张量作为黑色图片
                # black_image = torch.zeros((height, width, 3), dtype=torch.uint8)
                # # 将张量转换为 ndarray
                # black_image_np = black_image.numpy()
                # # 将 ndarray 转换为 PIL 图像
                # black_image_pil = Image.fromarray(black_image_np)
                # images = [black_image_pil]
                
                #infer without rag
                images_list = []
                for img in sample['images_list']:
                    images_list.append(args.image_file + img)
                
                if args.rag:
                    sample['context'], images_list = rag(sample['context'], images_list, question, 4000)
                    print('ragging')
                images = load_images(images_list)
                
                #infer with rag
                # sample['context'] = '[<IMG_PLH>]' + sample['context']
                # # 定义图片的大小
                # height, width = 256, 256
                # # 生成全零张量作为黑色图片
                # black_image = torch.zeros((height, width, 3), dtype=torch.uint8)
                # # 将张量转换为 ndarray
                # black_image_np = black_image.numpy()
                # # 将 ndarray 转换为 PIL 图像
                # black_image_pil = Image.fromarray(black_image_np)
                # images = [black_image_pil]
                
                #incontxt without rag
                # images_list = []
                # for img in sample['images_list']:
                #     index = img.find('save_long_aug_vqa_05_01_img_local_paths/')
                #     images_list.append(args.image_file + img[index+len('save_long_aug_vqa_05_01_img_local_paths/'):])
                # images = load_images(images_list)
                sample['context'] = re.sub(r'<image>', '[<IMG_PLH>]', sample['context'])
                query = sample['context'] + "Question:" + question
                
                # `[<IMG_PLH>]` is the image placeholder which will be replaced by image embeddings. 
                # the number of `[<IMG_PLH>]` should be equal to the number of input images
                # query = "[<IMG_PLH>][red, white, 3, bottom left].[<IMG_PLH>][yellow, white, 2, top left].[<IMG_PLH>][green, black, 4, bottom right].[<IMG_PLH>], what is the difference between these pictures?"

                # images = [
                #     Image.open("./examples/red_white_3_bottom_left.jpg").convert('RGB'),
                #     Image.open("./examples/yellow_white_2_top_right.jpg").convert('RGB'),
                #     Image.open("./examples/green_black_4_bottom_right.jpg").convert('RGB'),
                #     Image.open("./examples/blue_black_1_top_left.jpg").convert('RGB')
                # ]
                try:
                    inputs = model.build_input_ids(
                        text=[query],
                        tokenizer=tokenizer,
                        image=images

                    )

                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            image=inputs["image"].to(torch.bfloat16),
                            max_new_tokens=64,
                            length_penalty=-1)

                    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    outputs = output_text[0]
                except Exception as e:
                    print(e)
                    outputs = "None"
                
                print(outputs)
                torch.cuda.empty_cache()
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