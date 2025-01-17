# RAG ChatBot: 小菲减肥助手

本项目实现了一个基于LangChain和Gradio的RAG聊天机器人应用，专门为减肥和营养指导提供帮助。使用了```thenlper/gte-large-zh```
的向量嵌入模型，通过FAISS进行相似性搜索来检索相关的书籍内容。

## 项目特色

- 精选的一系列科学的减肥书籍作为知识库 :heart:
```
├── 饮食术2.epub
├── 健脑饮食.epub
├── 逆龄饮食.epub
├── 饮食的悖论.epub
├── 韩国饮食文化研究.epub
├── 放弃减肥，我瘦了60斤.epub
├── 我的最后一本减肥书.epub
├── 减脂生活：基础代谢减肥法.epub
├── 救命饮食：中国健康调查报告.epub
├── 饮食大脑：食物如何影响心理健康.epub
├── 疗愈的饮食与断食：新时代的个人营养学.epub
└── 饮食代谢力：不节食，不暴走，吃出易瘦体质.epub
```
- 所有的回答都将基于知识库来增强 :fire:
- 回答问题的同时，还能向用户推荐知识库中的书籍 :thumbsup:

## Demo

![demo](./docs/dojo_1.gif)

## 目录及文件说明

```
.
├── data                    知识库书籍，均为.epub格式
├── book_store              装有知识库所有书籍的本地向量数据库
├── make_book_store.py      读取data中的书籍并存入本地向量数据库的脚本
├── book_loader.py          根据langchain中的epub loader改良的loader
├── chat_bot.py             项目核心chat bot代码
├── app.py                  项目运行入口代码
└── requirements.txt        项目依赖文件
```

## 快速开始

以下是项目的主要文件和功能说明：

### 1.下载项目

```bash
git clone https://github.com/Blackoutta/slimmer_bot.git
```

### 2. 安装依赖

```bash
cd slimmer_bot
pip install -r requirements.txt
```

### 3. 运行项目

#### 参数说明

支持的命令行参数有:

- --api-key
- --base-url
- --model

api-key和base-url也可以通过直接设置环境变量来改变:

```
export OPENAI_API_KEY={your-key}
export OPENAI_BASE_URL={your-url}
```

```bash
python app.py --api-key {your-key} --base-url {your-url} --model {your-model}
```

#### 使用OPENAI

设置好OPENAI_API_KEY环境变量即可，默认使用`gpt-3.5-turbo`模型

```bash
python app.py
```

也可以通过命令行输入api-key

```bash
python app.py --api-key={your-key}
```

#### 使用智谱GLM-4

```bash
python app.py --api-key={your-key} --base-url https://open.bigmodel.cn/api/paas/v4/ --model glm-4
```

### 4. 访问服务

打开浏览器，访问:

```
0.0.0.0:7860
```
