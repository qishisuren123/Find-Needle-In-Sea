import io
import os
import re
import json
import hashlib
import torch
import random
import torchvision.transforms as T

from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from torchvision.transforms.functional import InterpolationMode
from petrel_client.client import Client

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

needle_find_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/data/needle-find.json'
needle_infer_pan_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/data/needle-infer-3-pan.json'
needle_img_path ='/mnt/petrelfs/renyiming/dataset/sea-needle/data/needle-find-v2.json'

def add_dict_to_json(new_data, json_file_path='/mnt/petrelfs/renyiming/dataset/sea-needle/data/ori_data.json'):
    # 检查文件是否存在，如果不存在，则创建一个空的JSON数组
    if not os.path.exists(json_file_path):
        with open(json_file_path, 'w') as file:
            json.dump([], file)
    
    # 读取现有数据
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # 添加新的字典到数组中
    data.append(new_data)
    
    # 写回更新后的数据到文件
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

def select_needle(file_path):
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Randomly select an element
    random_element = random.choice(data)
    
    return random_element

def select_img_needle(file_path):
    abnormal_pic_file = '/mnt/petrelfs/renyiming/dataset/sea-needle/abnormal_pic'
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Randomly select an element
    random_element = random.choice(data[:11])
    thing = random_element
    image_path = abnormal_pic_file + random_element['answer'] + '.jpg'
    
    return thing, image_path
    
def insert_mul_img_needle(sample, index, thing):
    sample['img_mul_needle'] = thing['sentence']
    sample['img_mul_question'] = thing['count_question']
    
    special_word = thing['sentence']
    article_list = sample['texts']
    sentence_split_regex = r'(?<=[.!]) (?=[A-Z])'
    sentences = []
    article_indices = []  

    for idx, text in enumerate(article_list):
        if text is not None:
            parts = re.split(sentence_split_regex, text)
            sentences.extend(parts)
            article_indices.extend([idx] * len(parts))

    total_sentences = len(sentences)
    if total_sentences > 3:
        insert_position = random.randint(0, total_sentences - 3)
        if insert_position + 1 < len(sentences):
            sentences.insert(insert_position + 1, special_word)
        else:
            sentences.append(special_word)  # Append if it's the last sentence
    elif total_sentences > 0:
        sentences.append(special_word)  # Append to the end if not enough sentences to choose from

    current_sentence = 0
    for i in range(len(article_list)):
        if article_list[i] is not None:
            num_sentences_in_article = article_indices.count(i)
            article_list[i] = ' '.join(sentences[current_sentence:current_sentence + num_sentences_in_article]).strip()
            current_sentence += num_sentences_in_article

    position_percentage = ((insert_position / total_sentences) * 100) if total_sentences > 0 else 0
    sample['img_mul_needle_location'] = position_percentage
    
    for image_index, _ in enumerate(sample['images']):
        if _ is not None:
            sample['images'][image_index] = f"/mnt/petrelfs/renyiming/dataset/sea-needle/img_mul_needle/{index:05d}/{image_index:03d}.jpg"

    return sample

def insert_find_needle(sample):
    needle = select_needle(needle_find_path)
    special_word = needle['sentence']
    article_list = sample['texts']

    # Define a regex to correctly split text into sentences, considering punctuation and ensuring the next word starts with an uppercase letter
    sentence_split_regex = r'(?<=[.!?]) (?=[A-Z])'

    # Split the articles into sentences and count them
    sentences = []
    article_indices = []  # Keep track of which article each sentence belongs to
    for idx, text in enumerate(article_list):
        if text is not None:
            # Splitting text into sentences using the defined regex
            parts = re.split(sentence_split_regex, text)
            sentences.extend(parts)
            article_indices.extend([idx] * len(parts))

    # Calculate the total number of sentences
    total_sentences = len(sentences)

    # Randomly choose a sentence after which to insert the special word
    insert_position = random.randint(0, total_sentences - 3)

    # Insert the special word after the chosen sentence
    if insert_position + 1 < len(sentences):
        sentences.insert(insert_position + 1, special_word)
    else:
        sentences.append(special_word)  # Append if it's the last sentence

    # Reconstruct the articles
    current_sentence = 0
    for i in range(len(article_list)):
        if article_list[i] is not None:
            # Collect the number of sentences for the current article
            num_sentences_in_article = article_indices.count(i)
            # Join the sentences back to form the updated article
            article_list[i] = ' '.join(sentences[current_sentence:current_sentence + num_sentences_in_article]).strip()
            # Move the current sentence index to the next article's first sentence
            current_sentence += num_sentences_in_article

    # Calculate the actual insertion position as a percentage of the total sentences
    position_percentage = (insert_position / total_sentences) * 100

    # Updating the sample dictionary with additional details
    sample['find_needle'] = special_word
    sample['find_question'] = needle['question']
    sample['find_answer'] = needle['answer']
    sample['find_needle_location'] = position_percentage

    return sample, needle['answer']



def insert_infer_needle(sample):
    needle = select_needle(needle_infer_pan_path)
    special_words = [needle['sentence1'], needle['sentence2'], needle['sentence3']]
    article_list = sample['texts']

    # 定义正则表达式以正确地分割句子，考虑标点符号并确保下一个单词以大写字母开头
    sentence_split_regex = r'(?<=[.!?])\s+(?=[A-Z])'

    # 分割文章到句子并追踪每个句子的原始文章索引
    sentences = []
    article_indices = []
    for idx, text in enumerate(article_list):
        if text is not None:
            parts = re.split(sentence_split_regex, text)
            sentences.extend(parts)
            article_indices.extend([idx] * len(parts))

    total_sentences = len(sentences)
    if total_sentences == 0:
        return sample  # 如果没有句子，直接返回

    # 确定每个部分的句子数量
    part_size = max(1, total_sentences // 3)  # 避免除以0
    parts_ranges = [(0, part_size), (part_size, 2 * part_size), (2 * part_size, total_sentences)]

    # 在每个部分的随机句子末尾插入特殊词汇
    try:
        for i, part_range in enumerate(parts_ranges):
            if part_range[0] < total_sentences:  # 确保范围开始小于句子总数
                insert_position = random.randint(*part_range)
                sentences[insert_position] += ' ' + special_words[i]
            else:  # 如果范围起点超出了句子总数
                sentences[-1] += ' ' + special_words[i]  # 将特殊词汇加到最后一个句子
    except IndexError as e:
        print(f"捕获 IndexError: {e}, 将剩余特殊词汇加到最后一个句子")
        sentences[-1] += ' ' + ' '.join(special_words[i:])  # 添加剩余的所有特殊词汇

    # 重构文章
    current_sentence = 0
    for i in range(len(article_list)):
        if article_list[i] is not None:
            num_sentences_in_article = article_indices.count(i)
            article_list[i] = ' '.join(sentences[current_sentence:current_sentence + num_sentences_in_article]).strip()
            current_sentence += num_sentences_in_article

    sample['infer_question'] = needle['question']
    sample['infer_answer'] = needle['answer']
    sample['infer_needle'] = special_words

    return sample


def insert_multiple_needle(sample, forbid_word=None):
    needle = select_needle(needle_find_path)
    while forbid_word == needle['answer']:
        needle = select_needle(needle_find_path)
    special_word = needle['sentence']
    article_list = sample['texts']
    
    # Define a regex to correctly split text into sentences, considering punctuation
    sentence_split_regex = r'(?<=[.!?])\s+(?=[A-Z])'
    
    # Split the articles into sentences and track their original article index
    sentences = []
    article_indices = []
    for idx, text in enumerate(article_list):
        if text is not None:
            parts = re.split(sentence_split_regex, text)
            sentences.extend(parts)
            article_indices.extend([idx] * len(parts))
    
    # Randomly determine the number of special words to insert (at most one per sentence)
    max_sentences_to_insert = len(sentences)
    words_to_insert = random.randint(1, max_sentences_to_insert)
    
    # Randomly select positions to insert the special words
    insert_positions = random.sample(range(max_sentences_to_insert), words_to_insert)
    
    # Insert the special word at the end of chosen sentences
    for pos in insert_positions:
        sentences[pos] += ' ' + special_word
    
    # Reconstruct the articles
    current_sentence_index = 0
    for i in range(len(article_list)):
        if article_list[i] is not None:
            num_sentences_in_article = article_indices.count(i)
            article_list[i] = ' '.join(sentences[current_sentence_index:current_sentence_index + num_sentences_in_article])
            current_sentence_index += num_sentences_in_article
    
    sample['mul_needle'] = special_word
    sample['mul_question'] = f'How many "{special_word}" are in the article?'
    sample['mul_answer'] = words_to_insert
    
    return sample


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def insert_image_at_random_position(main_img, insert_img_path='/mnt/petrelfs/renyiming/dataset/sea-needle/abnormal_pic/apple1.jpg', scale_ratio=0.25):
    # 加载要插入的图片
    insert_img = Image.open(insert_img_path).convert('RGBA')  # 确保图片带有Alpha通道
    # 确保插入图片的大小是主图片的1/10
    insert_size = int(min(main_img.size) * scale_ratio)
    # 缩放插入图片
    insert_img = insert_img.resize((insert_size, insert_size))

    # 生成随机插入位置
    max_x = main_img.width - insert_size
    max_y = main_img.height - insert_size
    random_x = random.randint(0, max_x)
    random_y = random.randint(0, max_y)

    # 如果插入的图片具有透明度，提取透明度通道作为掩码
    if insert_img.mode == 'RGBA':
        mask = insert_img.split()[3]  # 提取Alpha通道作为掩码
        main_img.paste(insert_img, (random_x, random_y), mask)
    else:
        main_img.paste(insert_img, (random_x, random_y))

    return main_img


def build_transform(is_train, input_size, pad2square=False, insert_img=False):
    if is_train:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.),
                                interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        if pad2square is False and insert_img is True:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.Lambda(lambda img: insert_image_at_random_position(img)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        elif pad2square is False and insert_img is False:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Lambda(lambda img: expand2square_and_insert_image(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                # T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    return transform

def build_transform_stage1(is_train, input_size, image_path, pad2square=False, insert_img=False):
    # 定义转换流程，但不包括ToTensor和Normalize
    transform_steps = [
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
    ]

    if is_train:
        transform_steps.append(T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=T.InterpolationMode.BICUBIC))
    
    if pad2square:
        transform_steps.append(T.Lambda(lambda img: expand2square_and_insert_image(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))))

    if insert_img:
        transform_steps.append(T.Lambda(lambda img: insert_image_at_random_position(img, image_path)))

    pre_transform = T.Compose(transform_steps)
    return pre_transform

def build_transform_stage2():
    # 定义转换流程，但不包括ToTensor和Normalize
    transform_steps = [
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]

    transform = T.Compose(transform_steps)
    return transform

def save_image(images, index):
    folder_path = f"/mnt/petrelfs/renyiming/dataset/sea-needle/img_mul_needle/{index:05d}"
    # 检查目标文件夹是否存在，如果不存在，则创建
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    for img_idx, image in enumerate(images):
        file_name = f"{img_idx:03d}.jpg"
        full_path = os.path.join(folder_path, file_name)
        image.save(full_path)
        print("save transformed image at:", full_path)
        


class InterleavedDataset(Dataset):
    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        num_image_token,
        image_size=224,
        is_train=False,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        use_img_start_end_token=True,
        max_num_images=6,
    ):
        super().__init__()
        data_path = meta['annotation']
        image_path = meta['root']

        self.template_name = template_name
        self.data_path = data_path
        self.image_path = image_path
        self.num_image_token = num_image_token
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        assert self.template_name.startswith('plain'), 'Only plain template is supported for pretraining with packed data.'
        assert not self.is_train, 'Data augmentation is unnecessary for pretraining with packed data.'

        self.tokenizer = tokenizer
        self.tcs_loader = tcs_loader
        self.transform = build_transform(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square)
        self.transform2 = build_transform(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square, insert_img=True)
        self.transform_stage1_org = build_transform_stage1(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square, image_path=None)
        self.transform_stage2 = build_transform_stage2()
        
        self.img_start_token_id = self.tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

        assert self.img_start_token_id != self.tokenizer.unk_token_id
        assert self.img_token_id != self.tokenizer.unk_token_id
        assert self.img_end_token_id != self.tokenizer.unk_token_id

        self.group_by_length = group_by_length
        assert not self.group_by_length, 'Group_by_length is unnecessary for pretraining with packed data.'

        # dynamic image (not supported yet)
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        assert not self.dynamic_image_size, 'Dynamic image size is not supported now.'

        self.use_img_start_end_token = use_img_start_end_token
        if self.use_img_start_end_token:
            self.image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * self.num_image_token}{IMG_END_TOKEN}'
        else:
            self.image_tokens = f'{IMG_CONTEXT_TOKEN * self.num_image_token}'
        # hyperparameters for interleaved documents
        self.max_num_images = max_num_images
        self.max_tokens = tokenizer.model_max_length
        print(f"{self.max_tokens=}")

        if os.path.isdir(data_path):
            self.shard_mode = True
            self.shards_length = json.load(open(os.path.join(data_path, "length.json"), "r"))
            self.shard_id_range = json.load(open(os.path.join(data_path, "shard_id_range.json"), "r"))
            self._length = sum(self.shards_length.values())
            self.check_shard_id_range(self.shard_id_range, self._length)
            for _shard_name, (start, end) in self.shard_id_range.items():
                if start == 0:
                    break
            # first_shard_name = list(self.shards_length.keys())[0]
            first_shard_name = _shard_name
            self.current_shard_name = first_shard_name
            with open(os.path.join(data_path, first_shard_name)) as file:
                self.current_shard_data = json.load(file)
            print(f"Initialize shard file to {self.current_shard_name}")
        else:
            self.shard_mode = False
            with open(data_path) as file:
                self.data = file.readlines()
            self._length = len(self.data)

    def __len__(self):
        return self._length

    @staticmethod
    def check_shard_id_range(shard_id_range, length):
        ids = []
        print(shard_id_range.values())
        for start, end in shard_id_range.values():
            ids.extend(range(start, end))
        assert sorted(ids)[:length] == list(range(0, length))

    def load_data(self, index):
        if self.shard_mode:
            start, end = self.shard_id_range[self.current_shard_name]
            # print("start:", start)
            # print("end", end)
            # print("index", index)
            if start <= index < end:

                return deepcopy(self.current_shard_data[index - start])

            for shard_name, (start, end) in self.shard_id_range.items():
                if start <= index < end:
                    self.current_shard_name = shard_name
                    with open(os.path.join(self.data_path, shard_name)) as file:
                        self.current_shard_data = json.load(file)
                    print(f"Change shard file to {self.current_shard_name}")
                    return deepcopy(self.current_shard_data[index - start])

        return deepcopy(self.data[index])

    def get_img_filename(self, web_url, imgmeta):
        if 'filename' in imgmeta:
            return imgmeta['filename']

        hash_object = hashlib.sha256(web_url.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

    def load_image(self, image_path_or_url):
        if "s3://" in self.image_path:
            # print("load_image:", image_path_or_url)
            return self.tcs_loader(image_path_or_url).convert("RGB")
        else:
            # load from local (or s3mount node)
            return Image.open(image_path_or_url).convert("RGB")

    def parse_sample(self, sample):
        images = sample["images"]
        # print("images:", images)
        texts = sample["texts"]
        metadata = sample["metadata"]
        metadata = json.loads(metadata)
        valid_image = sample.get("valid_image", [True] * sum(img is not None for img in images))

        assert len(images) == len(texts)

        num_images = sum(img is not None for img in images)
        num_placeholders = sum(txt in ["<image>", None] for txt in texts)
        assert num_images == num_placeholders == len(valid_image), f"{num_images=}, {num_placeholders=}, {len(valid_image)=}, {sample=}"

        for _img, _imgmeta in zip(images, metadata):
            assert( _img is None) == (_imgmeta is None), sample

        return images, texts, metadata, valid_image
    
    def random_transform(self, images, index, image_path):
        transform_stage1_ins = build_transform_stage1(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square, insert_img=True, image_path=image_path)
        # 统计使用 transform2 的次数
        count_transform2 = 0
        
        # 处理每张图片
        pixel_values = []
        for image in images:
            # 随机选择 transform 或 transform2
            chosen_transform = random.choice([self.transform_stage1_org, transform_stage1_ins])
            
            # 应用选中的转换函数
            transformed_image = chosen_transform(image)
            print("after transform stage1 size:", transformed_image.size)
            pixel_values.append(transformed_image)
            
            # 如果选中的是 transform2，更新计数器
            if chosen_transform == transform_stage1_ins:
                count_transform2 += 1
                
        save_image(pixel_values, index)
        result = []
        for image in pixel_values:
            result.append(self.transform_stage2(image))
        
        return result, count_transform2

    def __getitem__(self, index):
        # 'images', 'metadata', 'general_metadata', 'texts', 'doc_loc', 'valid_image'
        sample = self.load_data(index)

        sample, forbid_word = insert_find_needle(sample)
        sample = insert_infer_needle(sample)
        sample = insert_multiple_needle(sample, forbid_word)
        
        sample_key = ['find_needle', 'find_question', 'find_answer', 'find_needle_location',
                      'infer_needle', 'infer_question', 'infer_answer',
                      'mul_needle', 'mul_question', 'mul_answer'
        ]
        for one_key in sample_key:
            print(one_key+':', sample[one_key])
            print('\n')
            
        print(sample)
        # add_dict_to_json(sample)

        # parse sample and check
        images, texts, metadata, valid_image = self.parse_sample(sample)

        print(f"{sum(valid_image)=}, {self.max_num_images=}")
        images = [
            os.path.join(self.image_path, self.get_img_filename(img, imgmeta))
            for img, imgmeta in zip(images, metadata)
            if img is not None
        ]
        
        if sum(valid_image) > self.max_num_images:
            true_count = 0
            for i in range(len(valid_image)):
                if valid_image[i] is True:
                    true_count += 1
                    if true_count > self.max_num_images:
                        valid_image[i] = False

        images = [self.load_image(img) for img, valid in zip(images, valid_image) if valid]
        
        #insert image needle or not:
        # pixel_values = [self.transform(image) for image in images]
        thing, image_path = select_img_needle(needle_img_path)
        sample = insert_mul_img_needle(sample, index, thing)
        image_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/abnormal_pic/' + thing['answer'] + '.jpg'
        pixel_values, num_img_insert = self.random_transform(images, index=index, image_path=image_path)
        print("num_img_insert:", num_img_insert)
        sample['img_mul_answer'] = num_img_insert
        
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        # print('got num_patches')

        # padding to max_num_images
        # if len(images_tensors) < self.max_num_images:
        #     zero_padding = torch.zeros(
        #         (
        #             self.max_num_images - len(images_tensors),
        #             N_CHANNELS,
        #             images_tensors[0].shape[1],
        #             images_tensors[0].shape[2],
        #         ),
        #         dtype=torch.float,
        #     )
        #     images_tensors = torch.cat((images_tensors, zero_padding), dim=0)

        # preprocess and tokenize text
        # add in <image> and <eoc> tokens
        image_idx = 0
        for i in range(len(texts)):
            if texts[i] is None:
                if valid_image[image_idx]:
                    texts[i] = "<image>"
                image_idx += 1
        texts = [_ for _ in texts if _]
        text = " ".join(texts)
        text = f"{text}{self.tokenizer.eos_token}"
        text = text.replace("<image> ", "<image>\n").replace(" <image>", "\n<image>")

        # e.g.: replace <image><image><image> with <image>\n<image>\n<image>
        repl = lambda match: '\n'.join('<image>' for _ in range(match.group(0).count('<image>')))
        text = re.sub(r'(<image>)+', repl, text)
        text_after_replace = text.replace('<image>', self.image_tokens)

        self.tokenizer.padding_side = "right"
        input_ids = self.tokenizer(
            text_after_replace,
            max_length=self.max_tokens,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ).input_ids

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = input_ids.clone()
        labels = torch.where(labels == self.img_start_token_id, IGNORE_TOKEN_ID, labels)
        labels = torch.where(labels == self.img_token_id, IGNORE_TOKEN_ID, labels)
        labels = torch.where(labels == self.img_end_token_id, IGNORE_TOKEN_ID, labels)

        # TODO: ignore <s>
        # TODO: ignore </s> directly following the image
        # TODO: add <s></s> for each text segment?

        ret = dict(
            input_ids=input_ids[0],
            labels=labels[0],
            attention_mask=attention_mask[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert('RGB')

class TCSLoader(object):

    def __init__(self, conf_path, sc_config_key='sensecore'):
        print(f'[TCSLoader] config_path: {conf_path}')
        print('--> before Client(conf_path)')
        self.client = Client(conf_path)
        self.sc_config_key = sc_config_key
        print('--> after Client(conf_path)')

    def __call__(self, fn):
        img_value_str = self.client.get(fn)
        img = pil_loader(img_value_str)
        return img
