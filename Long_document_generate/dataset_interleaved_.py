import io
import os
import re
import json
import hashlib
import torch
import torchvision.transforms as T

from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from torchvision.transforms.functional import InterpolationMode
from petrel_client.client import Client
import yaml

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
                T.Lambda(lambda img: expand2square(img, tuple(int(x * 255) for x in (0.485, 0.456, 0.406)))),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    return transform

class InterleavedDataset(Dataset):
    def __init__(
        self,
        long_save_path,
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
        token_max = 1000,
    ):
        super().__init__()
        data_path = meta['annotation']
        image_path = meta['root']
        self.long_save_path = "/mnt/lustre/liushuo/VQAG/long_document"
        self.token_max = token_max
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
        
        document_path = os.path.join(self.long_save_path, str(self.token_max))    
        self._length = len(os.listdir(document_path))
    def __len__(self):
        return self._length

    @staticmethod
    def check_shard_id_range(shard_id_range, length):
        ids = []
        for start, end in shard_id_range.values():
            ids.extend(range(start, end))
        assert sorted(ids)[:length] == list(range(0, length))
    '''
    def load_data(self, index):
        if self.shard_mode:
            start, end = self.shard_id_range[self.current_shard_name]
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
    '''

    def load_data(self, index):
        document_path = os.path.join(self.long_save_path, str(self.token_max), '{}.yaml'.format(index))   
        with open(document_path, "r") as file:
            doc_sample = yaml.safe_load(file)

        return doc_sample



    def get_img_filename(self, web_url, imgmeta):
        if 'filename' in imgmeta:
            return imgmeta['filename']

        hash_object = hashlib.sha256(web_url.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig

    def load_image(self, image_path_or_url):
        if "s3://" in self.image_path:
            return self.tcs_loader(image_path_or_url).convert("RGB")
        else:
            # load from local (or s3mount node)
            return Image.open(image_path_or_url).convert("RGB")

    def parse_sample(self, sample):
        images = sample["images_ori"]
        texts = sample["texts"]
        metadata = sample["metadata"]
        valid_image = sample["valid_image"]
        
        return images, texts, metadata, valid_image

    def __getitem__(self, index):
        # 'images', 'metadata', 'general_metadata', 'texts', 'doc_loc', 'valid_image'
        # sample = json.loads(self.load_data(index))
        sample = self.load_data(index)

        # parse sample and check
        images_ori, texts, metadata, valid_image = self.parse_sample(sample)

        # get valid images
        images = [self.load_image(img) for img in images_ori]

        # preprocess and pad images
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

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
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            question = vqa_question,
            answer = vqa_answer
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

