import json
from operator import itemgetter
from typing import List

from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.memory import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.base import T
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


class ChatBot:
    def __init__(self, api_key: str, base_url: str, vector_store_dir="book_store", model='gpt-3.5-turbo'):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.vector_store_dir = vector_store_dir
        self.db = FAISS.load_local(vector_store_dir, HuggingFaceEmbeddings(model_name='thenlper/gte-large-zh'),
                                   allow_dangerous_deserialization=True)
        self.retriever = self.db.as_retriever(search_type="similarity_score_threshold",
                                              search_kwargs={
                                                  "k": 3,
                                                  "score_threshold": 0.3
                                              })
        self.llm = ChatOpenAI(model=self.model, temperature=0.6,
                              openai_api_key=self.api_key,
                              openai_api_base=self.base_url, verbose=True)
        self.output_parser = NormalOutputParser()
        self.history = ChatMessageHistory()

    def get_ctx(self, question: str) -> (str, List[dict]):
        """

        :param question: 用户问题
        :return: str->检索到的上下文的完整string, List[dict] -> 检索到的上下文字典
        """
        docs = self.retriever.invoke(question)
        js = []
        for doc in docs:
            book_name = doc.metadata['filename']
            js.append({"book_name": book_name, "context": doc.page_content})
        print(js)
        d = json.dumps(js, ensure_ascii=False)
        return d, js

    def add_user_history(self, msg: str) -> str:
        self.history.add_user_message(msg)
        return msg

    def chat(self, question: str) -> dict:
        """
        总体思路:

        根据聊天记录理解、提炼用户问题
                     |
                     |
                 查询知识库
                 |       |
               |            |
            |                  |
            |                   |
用知识库中的知识预回答         整理知识的出处
                     |
                     |
             将两种回答整理成最终回答

        :param question: 用户问题
        :return: {'result': chat_bot的最终回答, 'source_documents': 检索到的上下文记录}
        """

        question_ctx_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{raw_question}"),
                HumanMessagePromptTemplate.from_template(
                    """总结出我真正的问题是什么，只用输出这个问题。
                    如果刚才的消息不是一个问题，那么只用礼貌性地回答一下就好。
                    example:
                    AI: 世界上有十种程序员。
                    Human: 哪十种?
                    AI(这应该是你的输出): 世界上的十种程序员分别是哪些? 
                    Human: 你好啊
                    AI(这应该是你的输出): 你好，请问有什么可以帮您? 
                    """
                )
            ]
        )

        summary_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    """
                       根据检索到的上下文来这个回答问题: "{question}",
                        '''
                        {context}
                        '''
                    """
                )
            ]
        )
        recommendation_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                """
                   根据检索到的上下文，按下方example生成内容,
                    '''
                    {context}
                    '''
                    example:
                    '''
                    1. 《书名1》中的相关知识
                    2. 《书名2》中的相关知识
                    '''
                """
            )
        ])

        combine_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """你是小菲，专业的减肥指导师和营养师, 你总是用优雅、古典、智慧的语气来回答问题。
                不用刻意提及你在回答问题，一个回答中不要重复地说同样的事情"""
            ),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(
                """
                用下面三引号内的知识，回答这个问题: {question}
                '''
                {summary}

                {recommendation}
                '''
                在回答的最后要提及你参考的书籍
                """
            )
        ])

        ctx, source_docs = self.get_ctx(question)

        question_ctx_chain = (
                question_ctx_prompt
                | self.llm
                | self.output_parser
                | RunnableLambda(self.add_user_history)
        )

        summary_chain = (
                summary_prompt
                | self.llm
                | self.output_parser
        )

        recommendation_chain = (
                recommendation_prompt
                | self.llm
                | self.output_parser
        )

        combine_chain = (
                combine_prompt
                | self.llm
                | self.output_parser
        )

        parsed_question = question_ctx_chain.invoke({"raw_question": question, "history": self.history.messages},
                                                    config={'callbacks': [ConsoleCallbackHandler()]})

        full_chain = (
                {
                    "summary": summary_chain,
                    "recommendation": recommendation_chain,
                    "history": itemgetter("history"),
                    "question": itemgetter("question")
                }
                | combine_chain
        )

        response = full_chain.invoke({
            "question": parsed_question, "context": ctx, "history": self.history.messages},
            config={'callbacks': [ConsoleCallbackHandler()]})

        self.history.add_ai_message(response)

        return {
            "result": response,
            "source_documents": source_docs
        }


class NormalOutputParser(BaseOutputParser):
    def parse(self, text: str) -> T:
        return text
