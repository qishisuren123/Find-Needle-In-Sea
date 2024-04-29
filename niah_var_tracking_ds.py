import io
import os
import re
import json
import jsonlines
import random

from math import ceil
from PIL import Image
from copy import deepcopy
from typing import List, Tuple, Optional
from pathlib import Path

import torch
import torchvision.transforms as T
import tiktoken

from num2words import num2words
from torch.utils.data import Dataset

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
        max_image_num: int = None,  # not implemented yet
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

    def _format_needle(self, needle_type: str, needle_ori: dict, needle_name: str) -> dict:
        """Format needle to standard.
        Args:
            needle_type (str):
            needle_ori (dict): Origin needle dict for each type.
            needle_name (str):
        standard needle format:
        {
            'needle_type': str,
            'name': str,  # idx for current needle
            'answer': int | str,  # int for choose, str for open
            'question': str,
            'choices': List[str] | None,
            'choices_image_path': List[str] | None,
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
                      name=needle_name,
                      choices=None,
                      choices_image_path=None)
        if needle_type == 'visual-reasoning':
            needle['meta'] = deepcopy(needle_ori['meta'])  # {'subset': str}
            # handle question / answer / choices_image_path for each subset
            if needle['meta']['subset'] == 'Jigsaw':
                question_end_index = needle_ori['prompt'].find('\nSelect')
                # use prompt is better
                needle['question'] = needle_ori['prompt'][:question_end_index]
                """Jigsaw prompt
                Given the first image with the lower right corner missing, 
                can you tell which one of the second image or the third image is the missing part? 
                Imagine which image would be more appropriate to place in the missing spot. 
                You can also carefully observe and compare the edges of the images.
                # Select from the following choices.\n\n(A) the second image\n(B) the third image\n
                """
                needle['answer'] = 0 if 'A' in needle_ori['answer'] else 1
                if self.image_file_name_only:
                    needle['choices_image_path'] = [os.path.basename(needle_ori['images'][1]),
                                                    os.path.basename(needle_ori['images'][2])]
                else:
                    needle['choices_image_path'] = [needle_ori['images'][1],
                                                    needle_ori['images'][2]]

            elif needle['meta']['subset'] == 'Multi-view_Reasoning':
                question_end_index = needle_ori['prompt'].find(' Select')
                needle['question'] = needle_ori['prompt'][:question_end_index]
                """Multi-view_Reasoning prompt
                The images are frames from a video. 
                The video is shooting a static scene. 
                The camera is either moving clockwise (left) or counter-clockwise (right) around the object. 
                The first image is from the beginning of the video and the second image is from the end. 
                Is the camera moving left or right when shooting the video?
                # Select from the following options.\n(A) left\n(B) right
                """
                needle['answer'] = 0 if 'A' in needle_ori['answer'] else 1
            else:
                raise NotImplementedError

            needle['choices'] = deepcopy(needle_ori['choices'])
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
        standard needle format:
        {
            'needle_type': str,
            'name': str,  # idx for current needle
            'answer': int | str,  # int for choose, str for open
            'question': str,
            'choices': List[str] | None,
            'choices_image_path': List[str] | None,
            # str for texts, dict for images {'type': str, 'path': str, 'token_num': int, 'meta': dict}
            'needles': List[str | dict],
            'meta': dict | None,
        }
        """
        # only consider text tokens for depth calculation
        def find_nearest_period(insertion_point: int, tokens_context: list) -> int:
            """Find (previous)nearest period in tokens_context and avoid special case.
            Special cases:
            ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'No.']
            Args:
                insertion_point (int): Index for tokens_context list of the insert point.
                tokens_context (list): List of tokens.

            Returns:
                insertion_point (int): Index of period. If no period before insertion_point, return 0.
            """
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
                if pre_text.endswith('Dr') or pre_text.endswith('Mr') or \
                        pre_text.endswith('Mrs') or pre_text.endswith('Ms') or pre_text.endswith('No'):
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

    def _generate_prompt(self, context: str, retrieval_question: str) -> str:
        prompt = ('You are an intelligent AI assistant skilled in '
                  'answering user questions.\n'
                  'Please keep your answers concise and clear. Do '
                  'not talk about irrelevant topics or repeat '
                  'your answers.\nThe document '
                  f'given to you by the user is:\n\n {context}\n\n'
                  f'Now, the question is: {retrieval_question}')
        return prompt

    def _format_result(self, longdoc: str, longdoc_metas: dict, needles: dict, visualization: bool = False) -> dict:
        """Convert result to convensional format.

        Args:
            longdoc (str): Long document(inserted with needles, <image> as image).
            longdoc_metas (dict): {
                                       'image_list'  : List[dict],  # image_dict {'type': str, 'path': str, 'token_num': int, 'meta': dict}
                                       'texts_token' : int,  # texts token num for longdoc(replace '<image>' with '')
                                       'images_token': int  # sum image token num in image_list
                                  }
            needles (dict): {'needle_name_0': needle_0, 'needle_name_1': needle_1, ...}, main needle is the last one.
            visualization (bool): Return more infos for visualization. Defaults to False.

        Returns:
            result (dict): Result by conventional format as follows:
            {
                'image_list': List[str],  # only file name
                'context': str,  # <image> alt for images
                'question': str,  # no options/choices
                'answer: str | int,  # str for open questions, int for choice index
                'meta': {
                    'placed_depth': List[float] | float,  # [0,1]
                    'context_length': int,  # image tokens + text tokens
                    'context_length_text': int,
                    'num_images': int,
                    'needles': List[str],
                    'choices': List[str] | None,  # None for not choice question
                    'choices_image_path': List[str] | None,  # None for not image answer
                    'category': str
                }
            }
            # 'id' is updated ouside.
        """
        def modify_question_and_choices(question: str, choices: list, needle_type: str, needle_meta: dict, info) -> str:
            if needle_type == 'infer-choose':
                return question, choices
            elif needle_type == 'visual-reasoning':
                # info (list): Indexs of needle images in the longdoc, start from 0.
                if needle_meta['subset'] == 'Jigsaw':
                    first_idx = num2words(info[0] + 1, to="ordinal")  # >= 1
                    second_idx = num2words(info[1] + 1, to="ordinal")  # >= 2
                    third_idx = num2words(info[2] + 1, to="ordinal")  # >= 3
                    question = question.replace('third', third_idx)
                    question = question.replace('second', second_idx)
                    question = question.replace('first', first_idx)
                    choices[0] = choices[0].replace('second', second_idx)
                    choices[1] = choices[1].replace('third', third_idx)
                elif needle_meta['subset'] == 'Multi-view_Reasoning':
                    first_idx = num2words(info[0] + 1, to="ordinal")
                    second_idx = num2words(info[1] + 1, to="ordinal")
                    question = question.replace('second', second_idx)
                    question = question.replace('first', first_idx)
                else:
                    raise NotImplementedError
                return question, choices
            else:
                raise NotImplementedError
            return
        # 1. Update All Needle Depth in LongDoc
        needle_depth = dict()
        needles_list = list(needles.items())
        for (needle_name, needle_dict) in needles_list:
            needle_depth[needle_name] = []
            if needle_dict['needle_type'] == 'infer-choose':
                # find texts
                longdoc_pure_text = longdoc.replace(self.image_alt_sym, '')
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
            else:
                raise NotImplementedError

        # 2. Modify Question and Choices by Main Needle
        main_needle_name, main_needle_dict = needles_list[-1]
        question = None
        choices = None
        if self.main_needle_type == 'infer-choose':
            question = main_needle_dict['needle_format']['question']
            # choices/choices_image_path = None
        elif self.main_needle_type == 'visual-reasoning':
            # find all image idxs
            needle_image_idx_list = []
            image_list = longdoc_metas['image_list']
            for image_idx, image in enumerate(image_list):
                if image['meta'].get('needle_name', None) == main_needle_name:
                    needle_image_idx_list.append(image_idx)
            assert len(needle_image_idx_list) == len(main_needle_dict['needle_format']['needles'])
            question, choices = modify_question_and_choices(
                main_needle_dict['needle_format']['question'],
                main_needle_dict['needle_format']['choices'],
                'visual-reasoning',
                main_needle_dict['needle_format']['meta'],
                needle_image_idx_list)
        else:
            raise NotImplementedError

        # 3. Format Final Result
        result = dict()
        token_num_texts = len(self.tokenizer_counter.encode(
            longdoc.replace(self.image_alt_sym, '')))
        token_num_images = longdoc_metas['images_token']
        token_num_total = (token_num_texts + token_num_images)
        image_path_list = []
        if self.image_file_name_only:
            for image_dict in longdoc_metas['image_list']:
                image_path_list.append(os.path.basename(image_dict['path']))
        else:
            for image_dict in longdoc_metas['image_list']:
                image_path_list.append(image_dict['path'])
        # check image num
        assert longdoc.count(self.image_alt_sym) == len(image_path_list)
        result['images_list'] = image_path_list
        result['context'] = longdoc
        result['question'] = question
        result['answer'] = main_needle_dict['needle_format']['answer']
        result['meta'] = {
            # depth for needles in main needle
            'placed_depth': needle_depth[main_needle_name],
            'context_length': token_num_total,
            'context_length_text': token_num_texts,
            'num_images': len(image_path_list),
            'needles': main_needle_dict['needle_format']['needles'],
            'choices': choices,
            'choices_image_path': main_needle_dict['needle_format']['choices_image_path'],
            'category': self.main_needle_type
        }
        if visualization:
            result['needles_meta'] = needles
        return result

    def _format_visualization_result(self, conversation: dict) -> str:
        """Convert conversation dict to markdown str format.

        Args:
            conversation (dict): result from __getitem__(index, visualization=True)

        Returns:
            md_str (str): Markdown string. Format as follows:
                # Meta Info
                conversation['meta']
                # Needles:
                Needle_0
                Needle_1
                ...
                ---
                # Prompt:
                prefix + text + images + question
                # Answer:
                answer
        """
        # 1. Get Infos from Conversation
        needles = conversation['needles_meta']
        longdoc = conversation['context']  # str
        question = conversation['question']
        answer = conversation['answer']
        choices = conversation['meta']['choices']
        answer = choices[answer] if choices else answer
        image_path_list = conversation['images_list']
        # convert image path to rel path
        for i, image_path in enumerate(image_path_list):
            idx = longdoc.find(self.image_alt_sym)
            image_path = image_path.replace('/mnt/petrelfs/share_data/duanyuchen/datasets/BLINK_custom/',
                                            '../images/')
            longdoc = longdoc[:idx] \
                + f"\n\n![image]({image_path})\n\n" \
                + longdoc[idx+len(self.image_alt_sym):]
        
        # 2. Mark Needle Texts in Different Color, Gather All Needles
        haystack_needle_color = 'Blue'
        main_needle_color = 'Red'
        needles = list(needles.items())
        main_needle_name, _ = needles[-1]
        needle_strs = ''
        for i, (needle_name, needle) in enumerate(needles):
            # mark each needle
            needle_type = needle['needle_type']
            needle_strs += f'## Needle {i}: \n\n'
            if needle_type == 'infer-choose':
                for j, needle_sentence in enumerate(needle['needle_format']['needles']):
                    if needle_name == main_needle_name:
                        longdoc = longdoc.replace(
                            needle_sentence, f'\n<font color={main_needle_color}>{needle_sentence}</font>\n')
                    else:
                        longdoc = longdoc.replace(
                            needle_sentence, f'\n<font color={haystack_needle_color}>{needle_sentence}</font>\n')
                    needle_strs += f'{j + 1}. {needle_sentence}\n\n'
                needle_strs += f'Question: {needle["needle_format"]["question"]}\n\n'
                needle_strs += f'Answer: {needle["needle_format"]["answer"]}\n\n'
            elif needle_type == 'visual-reasoning':
                for j, image in enumerate(needle['needle_format']['needles']):
                    image_path = image['path'].replace('/mnt/petrelfs/share_data/duanyuchen/datasets/BLINK_custom/',
                                                       '../images/')
                    needle_strs += f"![image]({image_path})\n\n"
                needle_strs += f'Question: {needle["needle_ori"]["prompt"]}\n\n'
                needle_answer = needle["needle_ori"]["answer"]
                needle_strs += f'Answer: {needle_answer}\n\n'
            else:
                raise NotImplementedError
        
        # 3. Format the Markdown Str
        prompt = self._generate_prompt(longdoc, question)
        md_str = f'# Meta Info\n\n```python\n\n{conversation["meta"]}\n\n```\n\n' \
            f'\n\n# Needles:\n\n{needle_strs}\n\n' \
            '\n\n---\n\n' \
            f'\n\n# Prompt:\n\n{prompt}\n\n'\
            f'\n\n# Answer:\n\n{answer}\n\n'
        return md_str

    def __len__(self):
        return self.dataset_len

    def __visitem__(self, index) -> str:
        """Get visualization str for data[index]
        """
        conversation = self.__getitem__(index, True)
        return self._format_visualization_result(conversation)

    def __getitem__(self, index, visualization=False) -> dict:
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
        result = self._format_result(
            longdoc, longdoc_metas, current_needles, visualization)
        result['id'] = index
        return result


if __name__ == '__main__':
    # Save NeedleHaystackVarTrackingDataset
    # max_image_num = None
    # max_image_size = None
    # token_max = [1000, 2000, 3000, 5000, 9000, 15000]

    # save_dir = Path('output/niah')

    # save_dir.mkdir(exist_ok=True)
    # dataset_len = 250
    # depth_percent_max = 90
    # main_needle_type = 'visual-reasoning'
    # haystack_needle_types = 'infer-choose'
    # for token_m in token_max:
    #     file_name = f'{main_needle_type}_depth_{depth_percent_max}_token_{token_m}.jsonl'
    #     file_path = save_dir / file_name
    #     file_path.unlink(missing_ok=True)
    #     file_path.touch()
    #     dataset = NeedleHaystackVarTrackingDataset(
    #         token_max=token_m,
    #         main_needle_type=main_needle_type,
    #         haystack_needle_types=haystack_needle_types,
    #         depth_percent_max=depth_percent_max,
    #         dataset_len=dataset_len
    #     )
    #     for i in range(dataset_len):
    #         data = dataset[i]
    #         with jsonlines.open(str(file_path), 'a') as f:
    #             f.write(data)
    # main_needle_type = 'infer-choose'
    # haystack_needle_types = 'visual-reasoning'
    # for token_m in token_max:
    #     file_name = f'{main_needle_type}_depth_{depth_percent_max}_token_{token_m}.jsonl'
    #     file_path = save_dir / file_name
    #     file_path.unlink(missing_ok=True)
    #     file_path.touch()
    #     dataset = NeedleHaystackVarTrackingDataset(
    #         token_max=token_m,
    #         main_needle_type=main_needle_type,
    #         haystack_needle_types=haystack_needle_types,
    #         depth_percent_max=depth_percent_max,
    #         dataset_len=dataset_len
    #     )
    #     for i in range(dataset_len):
    #         data = dataset[i]
    #         with jsonlines.open(str(file_path), 'a') as f:
    #             f.write(data)
    
    # Save Visualization
    max_image_num = None
    max_image_size = None
    token_max = 5000
    token_max_type = 'text'
    file_num = 10
    longdoc_dir = './output/longdocs_with_path/'
    main_needle_type = 'visual-reasoning'
    haystack_needle_types = 'infer-choose'
    depth_percent_max = 90
    dataset_len = 20
    dataset = NeedleHaystackVarTrackingDataset(
        token_max=token_max,
        longdoc_dir=longdoc_dir,
        main_needle_type=main_needle_type,
        haystack_needle_types=haystack_needle_types,
        depth_percent_max=depth_percent_max,
        dataset_len=dataset_len,
        image_file_name_only=False
    )
    for i in range(dataset_len):
        with open(f'./output/visualization/{token_max}/{main_needle_type}_{i}.md', 'w') as f:
            f.write(dataset.__visitem__(i))
