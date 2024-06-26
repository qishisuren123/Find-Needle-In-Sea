import io
import json
import jsonlines
import hashlib

from math import ceil
from copy import deepcopy
from typing import List, Tuple, Union, Optional
from pathlib import Path
from urllib import request
from multiprocessing import Process

import torch
import tiktoken

from petrel_client.client import Client
from tqdm import tqdm

from utils_dyc import (HTTP_PROXY)

null = None


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
                 save_path: Optional[str] = './output/longdocs/',
                 from_scratch: bool = False
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
            from_scratch (bool): Remove old longdoc files.
        """
        print(f'## Init LongDocumentGenerateWrapper')
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
        print(f'## Tokenizer:        {tokenizer}')
        print(f'## Token Num max:    {self.token_max}')
        print(f'## Image Num max:    {self.max_image_num}')
        print(f'## Image Size max:   {self.max_image_size}')
        print(f'## Image Patch Size: {self.image_patch_size}')
        print(f'## LongDocs saved at {str(self.save_path)}')

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
            print('Invalid Text Path or Petrel Client')
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
        self.from_scratch = from_scratch

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

    def check_longdoc_file(self, file: str) -> int:
        # check unbroken lines in file
        valid_lines = 0
        longdoc_list = []
        try:
            with jsonlines.open(file, 'r') as reader:
                for longdoc in tqdm(reader, desc=f'checking {file}', unit='sample'):
                    longdoc_list.append(longdoc)
                    valid_lines += 1
            if valid_lines:
                print(f'All lines valid in {file}!')
        except Exception as e:
            print(f'{e} at line {valid_lines}, adding samples...')
            with jsonlines.open(file, 'w') as writer:
                writer.write_all(longdoc_list)
        return valid_lines

    def generate(self, return_samples: bool = False) -> None:
        print('## Start Generate...')
        doc_sample_list = [] if return_samples else None
        for token_max, file_num in zip(self.token_max, self.file_num):
            print(f'## Generating Max Token {token_max}')
            if self.save_path:
                doc_file_path = self.save_path / \
                    f'{token_max}_token_{self.token_max_type}_{file_num}_samples.jsonl'
                doc_file_path.parent.mkdir(exist_ok=True)
                if self.from_scratch:
                    doc_file_path.unlink(missing_ok=True)
                doc_file_path.touch(exist_ok=True)
            valid_lines = self.check_longdoc_file(str(doc_file_path))
            while valid_lines < file_num:
                for file_idx in tqdm(range(valid_lines, file_num), desc=f'{token_max} token', unit='sample', dynamic_ncols=True):
                    success = False
                    while not success:
                        try:
                            long_doc_sample = self.generate_long_doc_sample(
                                token_max)
                            if self.save_path:
                                with jsonlines.open(str(doc_file_path), 'a') as f:
                                    f.write(long_doc_sample)
                            if return_samples:
                                doc_sample_list.append(long_doc_sample)
                            success = True
                        except UnicodeEncodeError as e:
                            print(e)
                            print('Retrying...')
                        except Exception as e:
                            raise e
                valid_lines = self.check_longdoc_file(str(doc_file_path))
        print('## Complete!')
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
    token_max_list = [[15000], [2000, 9000], [1000, 3000, 5000]]
    token_max_type = 'text'
    file_num = 100
    process_list = []
    for token_max in token_max_list:
        process_list.append(Process(target=generate_longdoc, args=[
                            max_image_num, max_image_size, token_max, token_max_type, file_num]))
    [p.start() for p in process_list]
    [p.join() for p in process_list]
