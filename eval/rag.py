# from langchain.docstore.document import Document
# from langchain.chains import RetrievalQA
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# import os

# text = '''Our loaded document is over 42k characters long. This is too long to fit in the context window of many models. Even for those models that could fit the full post in their context window, models can struggle to find information in very long inputs.

# To handle this we’ll split the Document into chunks for embedding and vector storage. This should help us retrieve only the most relevant bits of the blog post at run time.

# In this case we’ll split our documents into chunks of 1000 characters with 200 characters of overlap between chunks. The overlap helps mitigate the possibility of separating a statement from important context related to it. We use the RecursiveCharacterTextSplitter, which will recursively split the document using common separators like new lines until each chunk is the appropriate size. This is the recommended text splitter for generic text use cases.

# We set add_start_index=True so that the character index at which each split Document starts within the initial Document is preserved as metadata attribute “start_index”.'''

# def rag(text, query, length=4096):
#     documents = Document(page_content=text)
#     top_k = length//100
    
#     text_splitter =RecursiveCharacterTextSplitter(
#         separators=["\n\n", "\n", "?", "."],
#         chunk_size=100,
#         chunk_overlap=0,
#     )
#     texts = text_splitter.split_documents([documents])
#     vectorstore = Chroma.from_documents(documents=texts, embedding=HuggingFaceEmbeddings())

#     # Retrieve and generate using the relevant snippets of the blog.
#     retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

#     retrieved_docs = retriever.invoke(query)

#     sim_doc = ''
#     for i in range(min(top_k, len(retrieved_docs))):
#         sim_doc += retrieved_docs[i].page_content
        
#     return sim_doc

# a = rag(text, 'who helps us?')
# print(a)

import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import re
from PIL import Image
from tools import get_input
from tqdm import tqdm

# 初始化CLIP模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_text(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.get_text_features(**inputs)
    return outputs

def encode_images(images):
    pil_images = [Image.open(image_path) for image_path in images]
    inputs = processor(images=pil_images, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs

def split_text(text, max_length=200):
    sentences = re.split(r'(?<=[。！？])', text)
    chunks = []
    current_chunk = ""
    current_length = 0
    for sentence in sentences:
        sentence_length = len(processor.tokenizer.encode(sentence))
        if current_length + sentence_length > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += sentence
            current_length += sentence_length
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def rag(text, images, query, n):
    # 将<image>占位符替换为图片列表中的图片
    split_texts = re.split(r'(<image>)', text)
    text_chunks = []
    img_indices = []
    for i, chunk in enumerate(split_texts):
        if chunk == "<image>":
            img_indices.append(len(text_chunks))
            text_chunks.append(chunk)
        else:
            split_chunks = split_text(chunk)
            text_chunks.extend(split_chunks)
    
    
    # 编码文本和图片
    text_features = encode_text([chunk for chunk in text_chunks if chunk != "<image>"])
    image_features = encode_images(images)
    
    # 编码查询
    query_features = encode_text([query])
    
    # 计算文本与查询的相似度
    text_similarities = cosine_similarity(query_features.detach().numpy(), text_features.detach().numpy()).flatten().tolist()
    
    # 计算图片与查询的相似度
    image_similarities = cosine_similarity(query_features.detach().numpy(), image_features.detach().numpy()).flatten().tolist()
    
    # 结合相似度，筛选最相关的文本和图片
    combined_similarities = []
    text_idx = 0
    image_idx = 0
    for chunk in text_chunks:
        if chunk == "<image>":
            combined_similarities.append((image_similarities[image_idx], chunk, image_idx, 'image'))
            image_idx += 1
        else:
            combined_similarities.append((text_similarities[text_idx], chunk, text_idx, 'text'))
            text_idx += 1
    
    # 根据相似度排序
    combined_similarities.sort(key=lambda x: x[0], reverse=True)
    
    # 选择总字符数不超过n的文本和图片，同时保持原始顺序
    selected_text = []
    selected_images = []
    total_length = 0
    selected_image_indices = set()
    selected_text_indices = set()

    for sim, chunk, idx, chunk_type in combined_similarities:
        if chunk_type == 'image':
            if total_length + 576 <= n:
                selected_image_indices.add(idx)
                total_length += 576
                print('after add image', total_length)
        else:
            chunk_length = len(processor.tokenizer.encode(chunk))
            if total_length + chunk_length <= n:
                selected_text_indices.add(idx)
                total_length += chunk_length
                print('after add text', total_length)
        
        
    print(total_length)

    for i, chunk in enumerate(text_chunks):
        if chunk == "<image>":
            if img_indices.index(i) in selected_image_indices:
                selected_text.append(chunk)
                selected_images.append(images[img_indices.index(i)])
        else:
            if i in selected_text_indices:
                selected_text.append(chunk)
    
    # 组合成连续的字符串
    final_text = "".join(selected_text)

    return final_text, selected_images
