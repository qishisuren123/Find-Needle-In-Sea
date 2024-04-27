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
    


def build_transform(is_train, input_size, pad2square=False):
    if is_train:
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.),
                                interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        if pad2square is False:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                # T.Lambda(lambda img: expand2square_and_insert_image(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    return transform

class InterleavedDataset(Dataset):
    def __init__(
        self,
        template_name,
        tokenizer,
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
        needle_path='/mnt/petrelfs/renyiming/dataset/sea-needle/eval/test2.jsonl',
        image_path = '/mnt/petrelfs/renyiming/dataset/sea-needle/abnormal_pic'
    ):
        super().__init__()

        self.needle_path = needle_path
        self.image_path = image_path
        self.template_name = template_name
        self.image_path = image_path
        self.num_image_token = num_image_token
        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        assert self.template_name.startswith('plain'), 'Only plain template is supported for pretraining with packed data.'
        assert not self.is_train, 'Data augmentation is unnecessary for pretraining with packed data.'

        self.tokenizer = tokenizer
        self.transform = build_transform(is_train=self.is_train, input_size=self.image_size, pad2square=self.pad2square)

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


    def __len__(self):
        return 0
    #TODO


    def load_data(self, index):
        cur_index = 0
        with open(self.needle_path, 'r') as file:
            for line in file:
                if cur_index == index:
                    sample = json.loads(line)
                    break
                else:
                    cur_index += 1

        return sample

    def load_image(self, image_path_or_url):
        if "s3://" in self.image_path:
            pass
        else:
            img_path = os.path.join(self.image_path, image_path_or_url)
            # load from local (or s3mount node)
            return Image.open(image_path_or_url).convert("RGB")

    def parse_sample(self, sample):
        images = sample["images_list"]
        texts = sample["conversations"][0]['value']
        metadata = sample["meta"]

        return images, texts, metadata

    def __getitem__(self, index):
        # 'images', 'metadata', 'general_metadata', 'texts', 'doc_loc', 'valid_image'
        sample = self.load_data(index)

        # parse sample and check
        images, texts, metadata = self.parse_sample(sample)

        images = [self.load_image(img) for img in images]
        # preprocess and pad images
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
    
        text = texts
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

