import os

import pypandoc
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from book_loader import BookLoader

"""
首先要安装pandoc，才能使用unstructured包里的loader
"""
pypandoc.download_pandoc()

"""
将data目录中的电子书切割好后存入向量数据库
"""

# 指定目录路径
directory_path = './data'

# 获取目录下所有文件和文件夹的名称
all_files_and_folders = os.listdir(directory_path)

# 过滤出仅包含文件的列表
all_files = [f for f in all_files_and_folders if os.path.isfile(os.path.join(directory_path, f))]

embeddings = (
    HuggingFaceEmbeddings(model_name='thenlper/gte-large-zh')
)
all_docs = []
for file in all_files:
    print("loading: " + file)
    docs = BookLoader(os.path.join(directory_path, file), mode="paged", book_name=file.strip(".epub")).load()
    print("splitting: " + file)
    split = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50).split_documents(docs)
    all_docs.extend(split)

print(f"embedding {len(all_docs)} documents...")

book_store = FAISS.from_documents(documents=all_docs, embedding=embeddings)
print("saving to local...")
book_store.save_local(folder_path='./book_store')
print('all done!')
