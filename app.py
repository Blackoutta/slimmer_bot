import argparse
import os
import sys
import time

import gradio as gr

from chat_bot import ChatBot

chat_bot: ChatBot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, help='api key')
    parser.add_argument('--base-url', type=str, help='base url')
    parser.add_argument('--model', default='gpt-3.5-turbo', type=str, help='model name')

    args = parser.parse_args()

    if args.api_key is None:
        api_key_from_env = os.getenv("OPENAI_API_KEY")
        if api_key_from_env is None:
            sys.exit("--api-key cli arg not set or OPENAI_API_KEY environment variable not set")

    if args.base_url is None:
        base_url_from_env = os.getenv("OPENAI_BASE_URL")
        if base_url_from_env is not None:
            args.base_url = base_url_from_env

    global chat_bot
    chat_bot = ChatBot(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key
    )
    launch_gradio()


def chat(message, history):
    print(f"[user message]{message}")

    global chat_bot
    ans = chat_bot.chat(message)
    print(f"[result]{ans['result']}")
    print(f"[source_documents]{ans['source_documents']}")

    # 伪stream
    res = ans["result"]
    for i in range(len(res)):
        time.sleep(0.02)
        yield res[:i + 1]


def launch_gradio():
    web_ui = gr.ChatInterface(
        fn=chat,
        title="小菲减肥DoJo",
        chatbot=gr.Chatbot(height=600),
    )

    web_ui.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    main()
