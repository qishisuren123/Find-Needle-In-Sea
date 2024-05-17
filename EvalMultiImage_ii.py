from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
import json

questype = "ii"

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


path = "/mnt/petrelfs/renyiming/LTT/model/InternVL-Chat-V1-5"
# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map='auto').eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
generation_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

# pixel_values1 = load_image('/mnt/petrelfs/renyiming/LTT/InternVL/examples/image1.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('/mnt/petrelfs/renyiming/LTT/InternVL/examples/image2.jpg', max_num=6).to(torch.bfloat16).cuda()
# pixel_shapes = [pixel_values1.shape[0], pixel_values2.shape[0]]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = "<image>和<image>是两张熊猫照片，请详细描述这两张图片" # Describe the two pictures in detail
# response, history = model.chatimagetext(tokenizer, pixel_values, question, generation_config, pixel_shapes = pixel_shapes, history=None, return_history=True)
# print(question, response)
json_path = '/mnt/petrelfs/renyiming/LTT/NeedleInSea/llava_style_new/llavastyle/' + questype + '.jsonl'
ans_file = open("/mnt/petrelfs/renyiming/LTT/InternVL/Eval/InternVL_" + questype + ".jsonl", "w")
file = open(json_path, 'r', encoding='utf-8')
sample = []
for line in file.readlines():
    dic = json.loads(line)
    sample.append(dic)

# for i in tqdm(range(len(sample))):
for i in tqdm(range(10)):
    pixel_shapes = []
    pixel_value = load_image("/mnt/petrelfs/renyiming/LTT/NeedleInSea/data/" + sample[i]["images_list"][0], max_num=6).to(torch.bfloat16).cuda()
    pixel_values = pixel_value
    pixel_shapes.append(pixel_value.shape[0])
    for j in range(1, len(sample[i]["images_list"])):
        pixel_value = load_image("/mnt/petrelfs/renyiming/LTT/NeedleInSea/data/" + sample[i]["images_list"][j], max_num=6).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values, pixel_value), dim=0)
        pixel_shapes.append(pixel_value.shape[0])
    for j in range(4):
        pixel_value = load_image("/mnt/petrelfs/renyiming/LTT/NeedleInSea/Picture/" + sample[i]['meta']["choices_image_path"][j], max_num=6).to(torch.bfloat16).cuda()
        pixel_values = torch.cat((pixel_values, pixel_value), dim=0)
        pixel_shapes.append(pixel_value.shape[0])
    question = sample[i]["context"] + '\n' + sample[i]["question"] + '\nA. <image>'  + '\nB. <image>' + '\nC. <image>' + '\nD. <image>' + "\nAnswer with the option's letter from the given choices directly." 
    # print(pixel_values.shape[0])
    try:
        response, history = model.chatimagetext(tokenizer, pixel_values, question, generation_config, pixel_shapes = pixel_shapes, history=None, return_history=True)
    except Exception as e:
        response = f"{e}"
    ans_file.write(json.dumps({ "index": sample[i]["id"],
                                "text": response,
                                "answer": sample[i]["answer"],
                                "needle_location": sample[i]["meta"]["placed_depth"],
                                "image_num": len(sample[i]["images_list"]),
                                "token_num": sample[i]["meta"]["context_length"]}) + "\n")

        

