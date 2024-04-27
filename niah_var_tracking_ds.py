import io
import os
import re
import json
import jsonlines
import hashlib
import random

from math import ceil
from PIL import Image
from copy import deepcopy
from typing import List, Tuple, Union, Optional
from pathlib import Path
from urllib import request
from multiprocessing import Process

import torch
import torchvision.transforms as T
import tiktoken

from num2words import num2words
from torch.utils.data import Dataset
from petrel_client.client import Client

from utils_dyc import (HTTP_PROXY)

null = None


class NeedleHaystackVarTrackingDataset(Dataset):
    def __init__(
        self,
        tokenizer_counter: str = 'gpt-4',
        longdoc_dir: str = './output/longdocs',
        longdoc_reuse_num: int = 5,
        needle_meta_dir: str = './metas',
        main_needle_type: str = 'visual-reasoning',
        haystack_needle_types: str | list | None = ['infer-choose'],
        needles_num_per_file: int = 3,
        token_max: int = 5000,
        depth_percent_max: int = 90,
        max_image_num: int = None,  # not implemented
        max_image_size: int = None,
        keep_ratio: bool = True,
        image_patch_size: int = 16,
        image_file_name_only: bool = True,
        image_alt_sym: str = '<image>',
        dataset_len: Optional[int] = None,
    ):
        """
        Args:
            tokenizer_counter (str): tiktoken tokenizer for count text tokens. Defaults to 'gpt-4'.
            longdoc_dir (str): Dir for long documents. Files should be like longdoc_dic / token_num / 0.json .Defaults to './output/longdocs'.
            longdoc_reuse_num (int): Number of samples share one long document. Defaults to 5.
            needle_meta_dir (str): Dir for meta files. Defaults to './metas'.
            main_needle_type (str): Needle type for Q&A. Defaults to 'visual-reasoning'.
            haystack_needle_types (str | list): Needle type as haystack. Defaults to ['infer-choose'].
            needles_num_per_file (int): (Main needle + Haystack needle) per file. Defaults to 3.
            token_max (int): Token num for long document. Defaults to 5000.
            depth_percent_max (int): Depth percenatage for main needle. Defaults to 90.
            max_image_num (int, optional): Max image num after insert needles. Defaults to None.
            max_image_size (int, optional): Max image size for needle images. Defaults to None.
            keep_ratio (bool, optional): Keep ratio when resize needle images. Defaults to True.
            image_patch_size (int): Defaults to 16.
            image_file_name_only (bool): Remove dir of image files in needle. Defaults to True.
            image_alt_sym (str): str alternate image in prompt. Defaults to '<image>'.
            dataset_len (Optional[int]): Defaults to None.
        """
        super().__init__()
        self.tokenizer_counter = tiktoken.encoding_for_model(tokenizer_counter)
        # 1. Manage LongDoc and Needles Load Params
        self.longdoc_dir = Path(longdoc_dir)
        self.longdoc_reuse_num = longdoc_reuse_num
        self.needles_meta_dir = Path(needle_meta_dir)
        self.main_needle_type = main_needle_type
        if haystack_needle_types:
            # ['infer-choose', 'visual-reasoning']
            self.haystack_needle_types = haystack_needle_types
            if not isinstance(self.haystack_needle_types, list):
                self.haystack_needle_types = [self.haystack_needle_types]
            assert self.main_needle_type not in self.haystack_needle_types
        elif needles_num_per_file > 1:
            raise AssertionError
        else:
            self.haystack_needle_types = None
        self.needle_types = (
            [self.main_needle_type] + self.haystack_needle_types) if self.haystack_needle_types else [self.main_needle_type]
        self.needles_num_per_file = needles_num_per_file
        self.depth_percent_max = depth_percent_max
        # LongDocs paths
        self.token_max = token_max
        self.longdoc_path_list = list((
            self.longdoc_dir / str(self.token_max)).iterdir())
        # Needle meta files
        # {needle_type_0: [needle_dict_0, ...], needle_type_1: [...], ...}
        self.needle_meta_files = dict()
        # {needle_type_0: int, needle_type_1: int, ...}
        self.needle_meta_files_num = dict()
        for needle_type in self.needle_types:
            with (self.needles_meta_dir / f'needle-{needle_type}.json').open('r') as f:
                self.needle_meta_files[needle_type] = json.load(f)
                self.needle_meta_files_num[needle_type] = len(
                    self.needle_meta_files[needle_type])

        # 2. Assign Needles to each LongDoc
        self.dataset_len = dataset_len if dataset_len else len(
            self.longdoc_path_list) * self.longdoc_reuse_num
        assert self.dataset_len <= len(
            self.longdoc_path_list) * self.longdoc_reuse_num
        """Needle list format
        [dict(
            main=main_needle,
            haystack=haystack_needle_list
            ), ...]
        """
        self.needle_list = self._random_assign_needles()

        # 3. Image Params
        self.max_image_num = max_image_num
        # only strict needle images now
        self.max_image_size = max_image_size
        self.keep_ratio = keep_ratio
        self.image_patch_size = image_patch_size
        self.image_file_name_only = image_file_name_only
        self.image_alt_sym = image_alt_sym

    def _random_assign_needles(self) -> List[dict]:
        """Random assign needles to each index.
        Assign one main needle and (self.needles_num_per_file - 1) haystack needles.
        Random set depth for haystack needles.
        Returns:
            assigned_needles (list): 
                assigned_needles[i] is the needles assigned to data[i]
                each is a dict:
                {
                    'main': {
                            'needle_type': str,
                            'index': int  # line num in needle meta file
                        }
                    'haystack': [
                        {
                            'needle_type': str,
                            'index': int,
                            'depth_percent': int  # depth percent random assigned for haystack needle
                        },
                        ...
                    ]
                }
            
        """
        main_type_needles_len = self.needle_meta_files_num[self.main_needle_type]
        assigned_needles = []  # List[dict]
        # {
        #     'main' : {'needle_type': str, 'index': int}
        #     'haystack': [{'needle_type': str, 'index': int, 'depth_percent': int}, {}, ...]
        # }
        # first assign main type needle to each file
        needle_indexs = torch.randint(0, main_type_needles_len, [
                                      self.dataset_len]).tolist()
        assigned_needles = [{'main': {'needle_type': self.main_needle_type,
                                      'index': idx},
                             'haystack': []} for idx in needle_indexs]
        if len(self.needle_types) > 1:
            # assign haystack needles to each file
            random_needle_type = torch.randint(0, len(self.haystack_needle_types), [
                                               self.dataset_len, self.needles_num_per_file - 1])
            for a_idx, assigned in enumerate(assigned_needles):
                haystack_needle_types = random_needle_type[a_idx, :].tolist()
                for h_needle_type_idx in haystack_needle_types:
                    h_needle_type = self.haystack_needle_types[h_needle_type_idx]
                    needle_idx = random.randint(
                        0, self.needle_meta_files_num[h_needle_type] - 1)
                    depth_percent = random.randint(10, 100)
                    assigned['haystack'].append({'needle_type': h_needle_type,
                                                 'index': needle_idx,
                                                 'depth_percent': depth_percent})
        return assigned_needles

    def _get_longdoc_file(self, index: int) -> dict:
        """Load LongDoc by index.
        Returns:
            longdoc_file (dict): 
            {
                'texts': List[str | None],
                'images': List[str | None],
                ...
                # See LongDocumentGenerateWrapper
            }
        """
        longdoc_path = self.longdoc_path_list[index // self.longdoc_reuse_num]
        longdoc_file = deepcopy(json.load(longdoc_path.open()))
        return longdoc_file

    def _join_longdoc(self, longdoc_file) -> Tuple[str, dict]:
        """Join texts by self.image_alt_sym, get important metas.
        image_dict format
        {
            'type': str,  # (belong to) 'hystack' or 'needle'
            'token_num': int  # image_token_num,
            'path': str,
            'meta': dict
        }
        Args:
            longdoc_file (dict): File object from self._get_longdoc_file
        Returns:
            longdoc (str): Texts
            longdoc_meta (dict): {
                'image_list': list,  # list of image_dict
                'texts_token': int,
                'images_token': int,
            }
        """
        texts = longdoc_file['texts']
        images = longdoc_file['images']
        metadata = longdoc_file['metadata']
        token_num = longdoc_file['token_num']
        longdoc = ''
        # List[dict] {'type': 'hystack', 'token_num': int, 'path': str, 'meta': dict}
        track_images = []
        track_text_tokens = 0
        track_image_tokens = 0
        for piece_idx in range(len(token_num)):
            if token_num[piece_idx] > 0:  # valid image or texts
                if texts[piece_idx]:
                    text_token_num = token_num[piece_idx]
                    longdoc += texts[piece_idx]
                    track_text_tokens += text_token_num
                elif images[piece_idx]:
                    image_token_num = token_num[piece_idx]
                    longdoc += '<image>'
                    track_images += [{'type': 'hystack',
                                      'token_num': image_token_num,
                                      'path': images[piece_idx],
                                      'meta': metadata[piece_idx]}]
                    track_image_tokens += image_token_num
                else:
                    raise AssertionError
        return longdoc, {'image_list': track_images,
                          'texts_token': track_text_tokens,
                          'images_token': track_image_tokens}

    def _format_needle(self, needle_type:str, needle_ori:dict, needle_name:str) -> dict:
        """Format needle to standard.
        Args:
            needle_type (str):
            needle_ori (dict): Origin needle dict for each type.
            needle_name (str):
        standard needle format:
        {
            'needle_type': str,
            'name': str,  # idx for current needle
            'answer': str,
            'question': str,
            # str for texts, dict for images {'type': str, 'path': str, 'token_num': int, 'meta': dict}
            'needles': List[str | dict],
            'meta': dict | None,
        }
        """
        def img_size_to_token_length(img_size: Tuple[int, int] | int) -> tuple:
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            long_side = max(img_size)
            if (self.max_image_size is not None) and (long_side > self.max_image_size):
                # need resize
                resize_ratio = self.max_image_size / long_side
                if self.keep_ratio:
                    H = round(img_size[0] * resize_ratio)
                    W = round(img_size[1] * resize_ratio)
                else:
                    H = min(img_size[0], self.max_image_size)
                    W = min(img_size[1], self.max_image_size)
            else:
                H = img_size[0]
                W = img_size[1]
            h = ceil(H / self.image_patch_size)
            w = ceil(W / self.image_patch_size)
            token_num = h * w
            return int(token_num), (int(H), int(W))

        # transfer to standard needle form
        needle = dict(needle_type=needle_type,
                      name=needle_name)
        if needle_type == 'visual-reasoning':
            needle['answer'] = needle_ori['answer']
            needle['question'] = needle_ori['prompt']  # use prompt is better
            needle['meta'] = needle_ori['meta']
            needle['needles'] = []
            """
                image_dict {
                    'type': str,
                    'path': str,
                    'token_num': int,
                    'meta': {'height': int, 'width': int, 'needle_name': str}
                }
            """

            for img_path in needle_ori['images']:
                img_dict = dict(path=img_path,
                                type='needle')
                img = T.ToTensor()(Image.open(img_path))
                img_size = (img.shape[1], img.shape[2])
                token_num, (H, W) = img_size_to_token_length(img_size)
                img_dict['token_num'] = token_num
                img_dict['meta'] = {'height': H,
                                    'width': W,
                                    'needle_name': needle_name}
                needle['needles'].append(img_dict)

        elif needle_type == 'infer-choose':
            needle['answer'] = needle_ori['answer']
            needle['question'] = needle_ori['question']
            needle['needles'] = [needle_ori['sentence1'],
                                 needle_ori['sentence2'],
                                 needle_ori['sentence3']]
            needle['meta'] = None
        else:
            raise NotImplementedError
        return needle

    def _insert_needle(self, needle: dict, longdoc: str, longdoc_metas: dict, depth_percent: int) -> Tuple[str | dict]:
        """
        standard needle:
        {
            'needle_type': str,
            'name': str,  # idx for current needle
            'answer': str,
            'question': str,
            # str for texts, dict for images {'path': str, 'token_num': int, 'meta': dict}
            'needles': List[str | dict],
            'meta': dict | None,
        }
        """
        # only consider text tokens for depth calculation
        def find_nearest_period(insertion_point: int, tokens_context: list):
            # tokens_context = self.tokenizer.encode(context)
            period_tokens = self.tokenizer_counter.encode('.')
            try:
                while (insertion_point > 0) and (tokens_context[insertion_point] not in period_tokens):
                    insertion_point -= 1
            except Exception as e:
                print('insertion_point:', insertion_point)
                print('len:', len(tokens_context))
                raise e
            if insertion_point > 0:
                # special cases
                pre_text = self.tokenizer_counter.decode(
                    tokens_context[:insertion_point])
                if pre_text.endswith('Dr') or pre_text.endswith('Mr') or pre_text.endswith('Mrs') or pre_text.endswith('Ms'):
                    return find_nearest_period(insertion_point - 1, tokens_context)
            return insertion_point

        def insert_image(image_in: dict, longdoc: str, longdoc_metas: dict, depth_percent: int) -> Tuple[str | dict]:
            """Insert image to longdoc at depth_percent.
                image_in
                {
                    'type': str
                    'path': str,
                    'token_num': int,
                    'meta': dict
                }
            """

            # find insert pos
            text_split_list = longdoc.split(self.image_alt_sym)
            token_list = []
            for idx, text in enumerate(text_split_list):
                if len(text) == 0:
                    token_list.append(0)
                else:
                    token_list.append(len(self.tokenizer_counter.encode(text)))
            token_total = sum(token_list)
            token_cumsum = torch.cumsum(torch.tensor(token_list), 0)
            token_num_depth = int(token_total * (1 - depth_percent / 100))
            insert_piece_idx = torch.sum(
                token_cumsum <= token_num_depth).item()
            longdoc_metas['image_list'] = \
                longdoc_metas['image_list'][:insert_piece_idx] \
                + [{'type': image_in['type'],
                    'token_num': image_in['token_num'],
                    'path': image_in['path'],
                    'meta': image_in['meta']}] \
                + longdoc_metas['image_list'][insert_piece_idx:]
            longdoc_metas['images_token'] += image_in['token_num']
            if insert_piece_idx >= 1:
                rest_token_num = token_num_depth - \
                    token_cumsum[insert_piece_idx - 1]
            else:
                rest_token_num = token_num_depth
            insert_piece = text_split_list[insert_piece_idx]
            insert_piece_token = self.tokenizer_counter.encode(insert_piece)
            period_idx = find_nearest_period(
                rest_token_num, insert_piece_token)
            if period_idx > 0:
                insert_piece_pre = self.tokenizer_counter.decode(
                    insert_piece_token[:period_idx + 1])
                insert_piece_post = self.tokenizer_counter.decode(
                    insert_piece_token[period_idx + 1:])
            else:
                insert_piece_pre = ''
                insert_piece_post = insert_piece
            text_split_list = text_split_list[:insert_piece_idx] + [
                insert_piece_pre, insert_piece_post] + text_split_list[insert_piece_idx + 1:]
            longdoc_new = self.image_alt_sym.join(text_split_list)
            return longdoc_new, longdoc_metas

        def insert_text(text_in: str, longdoc: str, longdoc_metas: dict, depth_percent: int) -> Tuple[str | dict]:
            # find insert pos
            text_split_list = longdoc.split(self.image_alt_sym)
            token_list = []
            for idx, text in enumerate(text_split_list):
                if len(text) == 0:
                    token_list.append(0)
                else:
                    token_list.append(len(self.tokenizer_counter.encode(text)))
            token_total = sum(token_list)
            token_cumsum = torch.cumsum(torch.tensor(token_list), 0)
            token_num_depth = int(token_total * (1 - depth_percent / 100))
            insert_piece_idx = torch.sum(
                token_cumsum <= token_num_depth).item()

            if insert_piece_idx >= 1:
                rest_token_num = token_num_depth - \
                    token_cumsum[insert_piece_idx - 1]
            else:
                rest_token_num = token_num_depth
            insert_piece = text_split_list[insert_piece_idx]
            insert_piece_token = self.tokenizer_counter.encode(insert_piece)
            period_idx = find_nearest_period(
                rest_token_num, insert_piece_token)
            if period_idx > 0:
                insert_piece_pre = self.tokenizer_counter.decode(
                    insert_piece_token[:period_idx + 1])
                insert_piece_post = self.tokenizer_counter.decode(
                    insert_piece_token[period_idx + 1:])
            else:
                insert_piece_pre = ''
                insert_piece_post = insert_piece
            text_split_list = text_split_list[:insert_piece_idx] + [
                insert_piece_pre + text_in + insert_piece_post] + text_split_list[insert_piece_idx + 1:]
            longdoc_new = self.image_alt_sym.join(text_split_list)
            longdoc_metas['texts_token'] += len(
                self.tokenizer_counter.encode(text_in))
            return longdoc_new, longdoc_metas

        needle_type = needle['needle_type']
        # insert text/image needles into longdoc
        if needle_type == 'infer-choose':
            depth_percent_list = [d for d in range(
                depth_percent, 0, -(depth_percent//len(needle['needles'])))]
            depth_percent_list = depth_percent_list[:len(needle['needles'])]
            assert len(depth_percent_list) == len(
                needle['needles']), f'list: {depth_percent_list}, dp:{depth_percent}, len:{len(needle["needles"])}'
            for needle_text, dp in zip(needle['needles'], depth_percent_list):
                longdoc, longdoc_metas = insert_text(
                    needle_text, longdoc, longdoc_metas, dp)
        elif needle_type == 'visual-reasoning':
            depth_percent_list = [d for d in range(
                depth_percent, 0, -(depth_percent//len(needle['needles'])))]
            depth_percent_list = depth_percent_list[:len(needle['needles'])]
            assert len(depth_percent_list) == len(
                needle['needles']), f'list: {depth_percent_list}, dp:{depth_percent}, len:{len(needle["needles"])}'
            for needle_image, dp in zip(needle['needles'], depth_percent_list):
                longdoc, longdoc_metas = insert_image(
                    needle_image, longdoc, longdoc_metas, dp)
        else:
            raise NotImplementedError
        return longdoc, longdoc_metas

    def _format_result(self, longdoc: str, longdoc_metas: dict, needles: dict) -> dict:
        def generate_prompt(context: str, retrieval_question: str) -> str:
            prompt = ('You are an intelligent AI assistant skilled in '
                      'answering user questions.\n'
                      'Please keep your answers concise and clear. Do '
                      'not talk about irrelevant topics or repeat '
                      'your answers.\nThe document '
                      f'given to you by the user is {context}\n\n'
                      f'Now, the question is: {retrieval_question}')
            return prompt

        def modify_question(question: str, needle_type:str, needle_meta: dict, info) -> str:
            if needle_type == 'infer-choose':
                return question
            elif needle_type == 'visual-reasoning':
                if needle_meta['subset'] == 'Multi-view_Reasoning':
                    text_list = question.split('.')
                    first_idx = num2words(info[0], to="ordinal")
                    second_idx = num2words(info[1], to="ordinal")
                    text_list[0] = (f'The {first_idx}'
                                    f' and {second_idx} images are frames from a video')
                    text_list[3] = text_list[3].replace('first', first_idx)
                    text_list[3] = text_list[3].replace('second', second_idx)
                    question = '.'.join(text_list)
                elif needle_meta['subset'] == 'Jigsaw':
                    text_list = question.split('?')
                    first_idx = num2words(info[0], to="ordinal")
                    second_idx = num2words(info[1], to="ordinal")
                    third_idx = num2words(info[2], to="ordinal")
                    text_list[0] = text_list[0].replace('first', first_idx)
                    text_list[0] = text_list[0].replace('second', second_idx)
                    text_list[0] = text_list[0].replace('third', third_idx)
                    text_list[1] = text_list[1].replace('second', second_idx)
                    text_list[1] = text_list[1].replace('third', third_idx)
                    question = '?'.join(text_list)
                return question
            else:
                raise NotImplementedError
            return
        # 1. Update All Needle Depth in LongDoc
        needle_depth = dict()
        needles_list = list(needles.items())
        for (needle_name, needle_dict) in needles_list:
            if needle_dict['needle_type'] == 'infer-choose':
                # find texts
                longdoc_pure_text = longdoc.replace(self.image_alt_sym, '')
                needle_depth[needle_name] = []
                for some_needle in needle_dict['needle_format']['needles']:
                    needle_pos = re.search(
                        some_needle, longdoc_pure_text)
                    if needle_pos is None:
                        save_dict = {'some_needle': some_needle,
                                     'needle_dict:': needle_dict,
                                     'longdoc:': longdoc}
                        with open('temp.json', 'w') as f:
                            json.dump(save_dict, f)
                    needle_pos = needle_pos.span()[0]
                    depth = 1 - len(self.tokenizer_counter.encode(longdoc_pure_text[:needle_pos])) \
                        / len(self.tokenizer_counter.encode(longdoc_pure_text))
                    needle_depth[needle_name].append(depth)
            elif needle_dict['needle_type'] == 'visual-reasoning':
                image_list = longdoc_metas['image_list']
                needle_depth[needle_name] = []
                for image_idx, image in enumerate(image_list):
                    if image['meta'].get('needle_name', None) == needle_name:
                        longdoc_text_split = longdoc.split(self.image_alt_sym)
                        longdoc_pure_text = longdoc.replace(
                            self.image_alt_sym, '')
                        longdoc_pre = ''.join(
                            longdoc_text_split[:image_idx + 1])
                        depth = 1 - len(self.tokenizer_counter.encode(longdoc_pre)) \
                            / len(self.tokenizer_counter.encode(longdoc_pure_text))
                        needle_depth[needle_name].append(depth)
        
        # 2. Format Prompt by Main Needle
        main_needle_name, main_needle_dict = needles_list[-1]
        if self.main_needle_type == 'infer-choose':
            prompt = generate_prompt(
                longdoc, main_needle_dict['needle_format']['question'])
        elif self.main_needle_type == 'visual-reasoning':
            # find all image idxs
            needle_image_idx_list = []
            image_list = longdoc_metas['image_list']
            for image_idx, image in enumerate(image_list):
                if image['meta'].get('needle_name', None) == main_needle_name:
                    needle_image_idx_list.append(image_idx)
            question = modify_question(main_needle_dict['needle_format']['question'],
                                       'visual-reasoning',
                                       main_needle_dict['needle_format']['meta'],
                                       needle_image_idx_list)
            prompt = generate_prompt(longdoc, question)
        
        # 3. Format Final Result
        result = dict()
        token_num_texts = len(self.tokenizer_counter.encode(
            prompt.replace(self.image_alt_sym, '')))
        token_num_images = longdoc_metas['images_token']
        token_num_total = (token_num_texts + token_num_images)
        image_path_list = []
        if self.image_file_name_only:
            for image_dict in longdoc_metas['image_list']:
                image_path_list.append(os.path.basename(image_dict['path']))
        else:
            for image_dict in longdoc_metas['image_list']:
                image_path_list.append(image_dict['path'])
        result['images_list'] = image_path_list
        result['conversations'] = [{"from": "human", "value": prompt}]
        # check image num
        assert len(prompt.split(self.image_alt_sym)) == (
            len(image_path_list) + 1)
        result['answer'] = main_needle_dict['needle_format']['answer']
        result['meta'] = {
            "placed_depth": needle_depth[main_needle_name],  # depth for needles in main needle
            "context_length": token_num_total,
            "context_length_text": token_num_texts,
            "context_length_image": token_num_images,
        }
        return result

    def __len__(self):
        return self.dataset_len

    def __visitem__(self, index):
        # TODO:
        raise NotImplementedError

    def __getitem__(self, index) -> dict:
        # 1. Get LongDoc
        longdoc_file = self._get_longdoc_file(index)
        longdoc, longdoc_metas = self._join_longdoc(longdoc_file)
        """ 
        longdoc: str  # <image> for image at that pos
        longdoc_metas: dict
        {'image_list': List[dict],
         'texts_token': int,
         'images_token': int}
        """
        # 2. Get Needles
        current_needles_meta = self.needle_list[index]
        """current_needles_meta format:
        {
            'main'    : {'needle_type': str, 'index': int},
            'haystack': [{'needle_type': str, 'index': int}, {}, ...]
        }
        index is the line number in needle meta file
        """
        current_needles = dict()
        """current_needles format:
            {
                needle_name_0: {'needle_type': str,
                                'needle_ori': dict,  # origin needle for each type
                                'needle_format': dict,  # format needle
                                'depth_percent': int},
                needle_name_1: dict,
                ...
            }
            needle_name format: f'{needle_type}_{line_num_in_meta_file}'
        """
        # updata haystack needles in current_needles
        for haystack_needle in current_needles_meta['haystack']:
            needle_type = haystack_needle['needle_type']
            needle_idx = haystack_needle['index']
            depth_percent = haystack_needle['depth_percent']
            needle_ori = self.needle_meta_files[needle_type][needle_idx]
            needle_name = f'{needle_type}_{needle_idx}'
            needle_format = self._format_needle(
                needle_type, needle_ori, needle_name)
            current_needles[needle_name] = {
                'needle_type': needle_type,
                'needle_ori': needle_ori,
                'needle_format': needle_format,
                'depth_percent': depth_percent
            }
        # updata main needle in current_needles
        main_needle_ori = self.needle_meta_files[
            current_needles_meta['main']['needle_type']][current_needles_meta['main']['index']]
        main_needle_name = f'{self.main_needle_type}_{current_needles_meta["main"]["index"]}'
        main_needle_format = self._format_needle(
            self.main_needle_type, main_needle_ori, main_needle_name)
        current_needles[main_needle_name] = {
            'needle_type': self.main_needle_type,
            'needle_ori': main_needle_ori,
            'needle_format': main_needle_format,
            'depth_percent': self.depth_percent_max
        }
        # 3. Insert Needles into LongDoc
        for needle_dict in current_needles.values():
            longdoc, longdoc_metas = self._insert_needle(needle_dict['needle_format'],
                                                         longdoc,
                                                         longdoc_metas,
                                                         needle_dict['depth_percent'])
        # 4. Format results
        result = self._format_result(longdoc, longdoc_metas, current_needles)
        result['id'] = index
        return result


class LongDocumentGenerateWrapper:
    """
    Long document json file format: (one json file for each document)
    {
        'texts'                : List[str | None],  # text -> str, image -> None
        'images'               : List[str | None],  # text -> None, image -> str(url | sha256)
        'valid_image'          : List[int],  # valid image -> 1, texts/invalid image -> 0
        'metadata'             : List[dict | None],  # text -> None, image -> dict
        'token_num_total'      : int,
        'text_token_num_total' : int,
        'image_token_num_total': int,
        'token_num'            : List[int],
        'token_num_text'       : List[int],
        'token_num_image'      : List[int],
        'token_max_type'       : str,  # 'all' or 'text'
    }

    Save format:
    save_path / 'token_max[0]' / '0.json'
                               / '1.json'
                               / ...
              / 'token_max[1]' / '0.json'
              ...
    """

    def __init__(self,
                 text_src_path: str = 's3://public-dataset/OBELISC/jsonl/',
                 image_src_path: Optional[str] = None,
                 tokenizer: str = 'gpt-4',
                 token_max: Union[int, List[int]] = 15000,
                 token_max_type: str = 'text',  # [text, all]
                 file_num: Union[int, List[int]] = 10,
                 max_image_num: Optional[int] = None,
                 max_image_size: Optional[int] = None,
                 keep_ratio: bool = True,
                 image_patch_size: int = 16,
                 petrel_config_path: str = '~/petreloss.conf',
                 petrel_cluster: str = 'obelisc',
                 hash_url: bool = True,
                 save_path: Optional[str] = './output/longdocs/'
                 ):
        """Wrapper for generate long documents.

        Args:
            text_src_path (str): Path to Obelisc dataset. Defaults to 's3://public-dataset/OBELISC/jsonl/'.
            image_src_path (Optional[str], optional): Set to None if read from urls in text files, else use local path. Defaults to None.
            tokenizer (str): tiktoken tokenizer. Defaults to 'gpt-4'.
            token_max (Union[int, List[int]]): Max tokens. Defaults to 15000.
            file_num (Union[int, List[int]]): File num for each token_max. Defaults to 10.
            max_image_num (int, optional): Max image num in each long document. Defaults to 6.
            max_image_size (int, optional): Max image size(long side) in long document. Defaults to 448.
            keep_ratio (bool): Keep origin ratio when resize images. Defaults to True.
            image_patch_size (int): Defaults to 16.
            petrel_config_path (str):  Defaults to '~/petreloss.conf'.
            petrel_cluster (str): Use your personal petreloss cluster. Defaults to 'obelisc'.
            hash_url (bool): Whether replace urls of images to sha256. Default to True.
            save_path (Optional[str], optional): save path for long documents. Defaults to './output/longdocs/'.
        """
        print(f'## Init LongDocumentGenerateWrapper', flush=True)
        # 1. Manage Generation Params
        self.text_src_path = text_src_path
        self.image_src_path = None
        if image_src_path:
            # image paths in text files are under this path
            self.image_src_path = Path(image_src_path)
        # Manage Tokenizer
        self.tokenizer = tiktoken.encoding_for_model(tokenizer)
        self.token_max = token_max if isinstance(
            token_max, list) else [token_max]
        self.token_max_type = token_max_type
        self.file_num = file_num if isinstance(file_num, list) else [
            file_num] * len(self.token_max)
        assert len(self.token_max) == len(self.file_num)
        # Manage Image Configs
        self.max_image_num = max_image_num
        self.max_image_size = max_image_size
        self.keep_ratio = keep_ratio
        self.image_patch_size = image_patch_size
        # Manage Save Path for Long Docs
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.hash_url = hash_url  # whether hash image urls
        print(f'## Tokenizer:        {tokenizer}', flush=True)
        print(f'## Token Num max:    {self.token_max}', flush=True)
        print(f'## Image Num max:    {self.max_image_num}', flush=True)
        print(f'## Image Size max:   {self.max_image_size}', flush=True)
        print(f'## Image Patch Size: {self.image_patch_size}', flush=True)
        print(f'## LongDocs saved at {str(self.save_path)}', flush=True)

        # 2. Manage File Loaders
        # Get file generator and check
        # text files: petrel backend
        self.petrel_client = Client(petrel_config_path)
        self.petrel_cluster = petrel_cluster
        self.petrel_cluster_header = self.petrel_cluster + ':s3://'
        # generator for text file path(the part after s3://)
        self.text_file_path_generator = self.petrel_client.get_file_iterator(
            f'{self.petrel_cluster}:{self.text_src_path}').__iter__()
        print(
            f'## LongDocs materials load from {self.petrel_cluster}:{self.text_src_path}')
        # track text file (first file is a test file)
        try:
            next(self.text_file_path_generator)
        except Exception as e:
            print('Invalid Text Path or Petrel Client', flush=True)
            raise e
        self.current_text_file_path = None  # str
        self.current_text_file = None  # io.BytesIO, current file
        self.current_text_file_sample = None  # dict, current sample
        # image files: local/http
        if self.image_src_path is None:  # http
            # install opener with personal proxy
            opener = request.build_opener(HTTP_PROXY)
            opener.addheaders = [
                ('User-agent',
                 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                 'Chrome/73.0.3683.86 Safari/537.36')]
            request.install_opener(opener)

    def _get_next_text_file(self) -> None:
        """
        Update [self.current_text_file_path, self.current_text_file] to next file.
        """
        try:
            self.current_text_file_path = f'{self.petrel_cluster}:s3://' + next(
                self.text_file_path_generator)[0]
            self.current_text_file = io.BytesIO(self.petrel_client.get(
                self.current_text_file_path))
        except StopIteration:
            print(f'Error: Not enough text files in {self.text_src_path} !!!')
            raise StopIteration

    def _get_next_text_sample(self) -> None:
        """
        Update self.current_text_file_sample to next line.
        """
        if self.current_text_file:
            sample = self.current_text_file.readline().decode('utf-8')
            if sample:
                self.current_text_file_sample = json.loads(sample)
                # str -> list
                self.current_text_file_sample['metadata'] = eval(
                    self.current_text_file_sample['metadata'])
                return
        self._get_next_text_file()
        self._get_next_text_sample()

    def _text_to_token_length(self, texts: list | str) -> list | int:
        """Convert str or List[str | None] to token nums.

        Args:
            texts (list | str): str or [str, None, ...]
        Returns:
            list | int: None -> 0, str -> token_num
        """
        if isinstance(texts, list):
            # text -> token len
            # None -> 0
            total_length = []
            for text_part in texts:
                if text_part:
                    total_length.append(len(self.tokenizer.encode(text_part)))
                else:
                    total_length.append(0)
        elif isinstance(texts, str):
            total_length = self.tokenizer.encode(texts)
        else:
            raise TypeError
        return total_length

    def _img_size_to_token_length(self, img_size: Tuple[int, int] | int) -> tuple:
        """Count image tokens (resize by self.max_image_size if set)

        Args:
            img_size (Tuple[int, int] | int): 

        Returns:
            tuple: token_num, (H_after_resize, W_after_resize)
        """
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        long_side = max(img_size)
        if (self.max_image_size is not None) and (long_side > self.max_image_size):
            # need resize
            resize_ratio = self.max_image_size / long_side
            if self.keep_ratio:
                H = round(img_size[0] * resize_ratio)
                W = round(img_size[1] * resize_ratio)
            else:
                H = min(img_size[0], self.max_image_size)
                W = min(img_size[1], self.max_image_size)
        else:
            H = img_size[0]
            W = img_size[1]
        h = ceil(H / self.image_patch_size)
        w = ceil(W / self.image_patch_size)
        token_num = h * w
        return int(token_num), (int(H), int(W))

    def _handle_images_and_metadatas(self, images: list, metadatas: list) -> Tuple[list]:
        """Check image url, count image tokens, update resize to imagemetas.

        Args:
            images (list): List of image url or None.
            metadatas (list): List of imagemeta. {'orign_height':int, ...}

        Returns:
            valid_images (list)
            token_length (list): None -> 0
            metadatas_new (list): metadatas[i].update({'height':int, 'width':int})
        """
        def _get_valid_imgs(img_paths: list) -> list:
            """Check image urls

            Args:
                img_paths (list): List of image url.

            Returns:
                list: None/invalid url -> 0, valid url -> 1 . 
            """
            valid_imgs = []
            if self.image_src_path:
                raise NotImplementedError
                for idx, img_path in enumerate(img_paths):
                    if img_path:
                        if (self.image_src_path / img_path).exists:
                            valid_imgs.append(1)
                        else:
                            valid_imgs.append(0)
                    else:
                        valid_imgs.append(0)
            else:
                for idx, img_path in enumerate(img_paths):
                    # valid img -> 1
                    # invalid img -> 0
                    # None -> 0
                    if img_path:
                        try:
                            request.urlopen(img_path, timeout=3)
                            valid_imgs.append(1)
                        except Exception as e:
                            valid_imgs.append(0)
                    else:
                        valid_imgs.append(0)
            return valid_imgs

        def _update_img_size_list_to_token_length(img_size_list: list) -> tuple:
            # resize images and calculate token nums
            token_length = []
            img_size_list_new = []
            for img_size in img_size_list:
                if img_size:
                    token_num, img_size_new = self._img_size_to_token_length(
                        img_size)
                    img_size_list_new.append(img_size_new)
                    token_length.append(token_num)
                else:
                    img_size_list_new.append(None)
                    token_length.append(0)
            return token_length, img_size_list_new

        def _update_img_metadatas(metadatas_old: list, img_size_list_new: list) -> list:
            metadatas_new = []
            for idx, metadata in enumerate(metadatas_old):
                if metadata:
                    metadata_new = deepcopy(metadata)
                    metadata_new['height'] = img_size_list_new[idx][0]
                    metadata_new['width'] = img_size_list_new[idx][1]
                    metadatas_new.append(metadata_new)
                else:
                    # set invalid image metas to None
                    metadatas_new.append(None)
            return metadatas_new

        valid_images = _get_valid_imgs(images)
        img_size_list = [((metadata['original_height'], metadata['original_width'])
                         if metadata else None) for metadata in metadatas]
        token_length, img_size_list_new = _update_img_size_list_to_token_length(
            img_size_list)
        metadatas_new = _update_img_metadatas(
            metadatas, img_size_list_new)
        return valid_images, token_length, metadatas_new

    def generate_long_doc_sample(self, token_max: int) -> dict:
        def hash_url(web_url):
            if web_url:
                hash_object = hashlib.sha256(web_url.encode())
                hex_dig = hash_object.hexdigest()
                return hex_dig
            else:
                return None

        token_num_list = torch.empty([0])
        token_num = 0
        token_num_judge = 0
        text_token_num_list = torch.empty([0])
        image_token_num_list = torch.empty([0])
        texts = []
        images = []
        valid_image = []
        metadata = []
        while token_num_judge < token_max:
            self._get_next_text_sample()
            # 1. Count Tokens for Current Sample
            current_texts_token_num_list = self._text_to_token_length(
                self.current_text_file_sample['texts'])
            current_valid_images_list, current_image_token_num_list, current_metadatas = self._handle_images_and_metadatas(
                self.current_text_file_sample['images'], self.current_text_file_sample['metadata'])
            # check length
            assert len(current_texts_token_num_list) == len(
                current_image_token_num_list)
            assert len(current_valid_images_list) == len(current_metadatas)
            assert len(current_texts_token_num_list) == len(
                self.current_text_file_sample['texts'])
            assert len(current_valid_images_list) == len(
                self.current_text_file_sample['images'])

            # 2. Add Texts & Images & Metas
            current_texts = deepcopy(self.current_text_file_sample['texts'])
            current_images = deepcopy(self.current_text_file_sample['images'])
            texts.extend(current_texts)  # str for texts, None for images
            images.extend(current_images)  # str for images, None for texts
            # None for texts, dict for images
            metadata.extend(current_metadatas)
            # 1 for valid image, 0 for invalid/texts
            valid_image.extend(current_valid_images_list)

            # check valid images num
            valid_images_tensor = torch.tensor(valid_image)
            if (self.max_image_num is not None) and (sum(valid_image) > self.max_image_num):
                # set extra images to invalid
                image_num_tensor = torch.cumsum(valid_images_tensor, dim=0)
                valid_images_tensor *= (image_num_tensor <= self.max_image_num)
                valid_image = valid_images_tensor.int().tolist()

            # 3. Clip & Break
            # concate token_num_list
            current_texts_token_num = torch.tensor(
                current_texts_token_num_list)
            current_image_token_num = torch.tensor(
                current_image_token_num_list)
            text_token_num_list = torch.cat(
                [text_token_num_list, current_texts_token_num])
            image_token_num_list = torch.cat(
                [image_token_num_list, current_image_token_num])
            image_token_num_list = image_token_num_list * valid_images_tensor
            token_num_list = text_token_num_list + image_token_num_list
            # +1 for final token_sum >= token_max
            if self.token_max_type == 'all':
                valid_tokens_num = torch.sum(
                    (torch.cumsum(token_num_list, dim=0) < token_max)).item() + 1
            elif self.token_max_type == 'text':
                valid_tokens_num = torch.sum(
                    (torch.cumsum(text_token_num_list, dim=0) < token_max)).item() + 1
            else:
                raise NotImplementedError
            texts = texts[:valid_tokens_num]
            images = images[:valid_tokens_num]
            valid_image = valid_image[:valid_tokens_num]
            metadata = metadata[:valid_tokens_num]
            token_num_list = token_num_list[:valid_tokens_num]
            token_num = torch.sum(token_num_list).int().item()
            text_token_num_list = text_token_num_list[:valid_tokens_num]
            image_token_num_list = image_token_num_list[:valid_tokens_num]
            # determine token num for judge
            if self.token_max_type == 'all':
                token_num_judge = token_num
            elif self.token_max_type == 'text':
                token_num_judge = torch.sum(text_token_num_list).int().item()
            else:
                raise NotImplementedError
        text_token_num_total = torch.sum(text_token_num_list).int().item()
        image_token_num_total = torch.sum(image_token_num_list).int().item()
        # convert urls to sha256
        if self.hash_url:
            images = list(map(hash_url, images))
        ret = dict(
            texts=texts,
            images=images,
            valid_image=valid_image,
            metadata=metadata,
            token_num_total=token_num,
            text_token_num_total=text_token_num_total,
            image_token_num_total=image_token_num_total,
            token_num=token_num_list.int().tolist(),
            token_num_text=text_token_num_list.int().tolist(),
            token_num_image=image_token_num_list.int().tolist(),
            token_max_type=self.token_max_type
        )
        return ret

    def generate(self, return_samples: bool = False) -> None:
        print('## Start Generate...', flush=True)
        doc_sample_list = [] if return_samples else None
        for token_max, file_num in zip(self.token_max, self.file_num):
            print(f'## Generating Max Token {token_max}', flush=True)
            for file_idx in range(file_num):
                long_doc_sample = self.generate_long_doc_sample(token_max)
                if self.save_path:
                    doc_file_path = self.save_path / \
                        str(token_max) / f'{file_idx}.json'
                    doc_file_path.parent.mkdir(exist_ok=True)
                    doc_file_path.touch(exist_ok=True)
                    with doc_file_path.open('w') as f:
                        json.dump(long_doc_sample, f, indent=11)
                if return_samples:
                    doc_sample_list.append(long_doc_sample)
        print('## Complete!', flush=True)
        return doc_sample_list


def generate_longdoc(max_image_num, max_image_size, token_max, token_max_type, file_num):
    long_doc_generate_wrapper = LongDocumentGenerateWrapper(
        max_image_num=max_image_num,
        max_image_size=max_image_size,
        token_max=token_max,
        token_max_type=token_max_type,
        file_num=file_num,
    )
    long_doc_generate_wrapper.generate(return_samples=False)


if __name__ == '__main__':
    # generate long documents
    max_image_num = None
    max_image_size = None
    token_max_list = [[1000, 15000], [2000, 9000], [3000, 5000]]
    token_max_type = 'text'
    file_num = 100
    process_list = []
    for token_max in token_max_list:
        process_list.append(Process(target=generate_longdoc, args=[
                            max_image_num, max_image_size, token_max, token_max_type, file_num]))
    [p.start() for p in process_list]
    [p.join() for p in process_list]

    # Save sNeedleHaystackVarTrackingDataset
    max_image_num = None
    max_image_size = None
    token_max = [1000, 2000, 3000, 5000, 9000, 15000]

    save_dir = Path('output/niah')

    save_dir.mkdir(exist_ok=True)
    dataset_len = 250
    depth_percent_max = 90
    main_needle_type = 'visual-reasoning'
    haystack_needle_types = 'infer-choose'
    for token_m in token_max:
        file_name = f'{main_needle_type}_depth_{depth_percent_max}_token_{token_m}.jsonl'
        file_path = save_dir / file_name
        file_path.unlink(missing_ok=True)
        file_path.touch()
        dataset = NeedleHaystackVarTrackingDataset(
            token_max=token_m,
            main_needle_type=main_needle_type,
            haystack_needle_types=haystack_needle_types,
            depth_percent_max=depth_percent_max,
            dataset_len=dataset_len
        )
        for i in range(dataset_len):
            data = dataset[i]
            with jsonlines.open(str(file_path), 'a') as f:
                f.write(data)
    main_needle_type = 'infer-choose'
    haystack_needle_types = 'visual-reasoning'
    for token_m in token_max:
        file_name = f'{main_needle_type}_depth_{depth_percent_max}_token_{token_m}.jsonl'
        file_path = save_dir / file_name
        file_path.unlink(missing_ok=True)
        file_path.touch()
        dataset = NeedleHaystackVarTrackingDataset(
            token_max=token_m,
            main_needle_type=main_needle_type,
            haystack_needle_types=haystack_needle_types,
            depth_percent_max=depth_percent_max,
            dataset_len=dataset_len
        )
        for i in range(dataset_len):
            data = dataset[i]
            with jsonlines.open(str(file_path), 'a') as f:
                f.write(data)
