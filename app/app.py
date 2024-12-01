import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import re
import os
import asyncio

# URLを検出する関数
def extract_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

# Webページの内容を取得する関数
def get_webpage_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text[:5000]
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"

# 検索クエリ生成のためのプロンプト
QUERY_PROMPT = """
あなたは与えられた質問に対して、以下の3つの判断を行うアシスタントです：
1. 最新の情報が必要かどうか
2. URLが含まれているかどうか
3. 通常の会話で対応可能かどうか

質問: {question}

以下の形式で応答してください：
NEEDS_SEARCH: [true/false] - 最新の情報が必要な場合はtrue
HAS_URL: [true/false] - URLが含まれている場合はtrue
SEARCH_QUERY: [検索クエリ] - NEEDS_SEARCHがtrueの場合のみ必要な検索クエリを書いてください
"""

def init_session_state():
    """セッション状態の初期化"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatGroq(
            model_name="llama-3.1-70b-versatile",
            temperature=0.7,
        )

async def handle_query(prompt, query_chain, search, extract_urls, get_webpage_content):
    try:
        # 質問の分析
        analysis = await query_chain.ainvoke(prompt)
        #st.markdown(analysis)
        content = analysis.content if hasattr(analysis, 'content') else str(analysis)
        needs_search = "NEEDS_SEARCH: true" in content
        has_url = "HAS_URL: true" in content
        #st.markdown(needs_search)

        # 応答生成
        if has_url:
            urls = extract_urls(prompt)
            if urls:
                webpage_content = get_webpage_content(urls[0])
                prompt_with_content = f"以下のWebページの内容に基づいて適切な返答を考えてください。広告や関連記事などに気を取られないでください。\n\nWebページ内容: {webpage_content}\n\n質問: {prompt}"
                response = await st.session_state.llm.apredict(prompt_with_content)
        elif needs_search:
            st.markdown("""DuckDuckGoで検索中...""")
            search_query = re.search(r'SEARCH_QUERY: (.*)', content)
            if search_query:
                search_results = search.run(search_query.group(1))
                prompt_with_search = f"""以下の検索結果の内容に基づいて適切な返答を考えてください。広告や関連記事などに気を取られないでください。
                できるだけ最新の情報を含めて回答してください。

                検索結果: {search_results}

                質問: {prompt}
                """
                # response = await llm.ainvoke(messages)
                # response = LangTools.sanitize_breakrow(response.content)
                #st.markdown(prompt_with_search)
                response = await st.session_state.llm.apredict(prompt_with_search)
            else:
                response = "申し訳ありません。検索クエリの生成に失敗しました。"
        else:
            chain = ConversationChain(
                llm=st.session_state.llm,
                memory=st.session_state.memory,
                verbose=True
            )
            response = await chain.arun(prompt)

        # 応答の表示
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")

def main():
    # Streamlitアプリの設定
    st.set_page_config(
        page_title="Enhanced Groq LangChain Chat",
        page_icon=":speech_balloon:",
        layout="wide"
    )
    st.title("Enhanced Groq LangChain Chat")

    # セッション状態の初期化
    init_session_state()

    # DuckDuckGo検索の初期化
    search = DuckDuckGoSearchAPIWrapper()

    # 質問分析のためのチェーン
    query_prompt = PromptTemplate(template=QUERY_PROMPT, input_variables=["question"])
    query_chain = query_prompt | st.session_state.llm
    #query_chain = LLMChain(llm=st.session_state.llm, prompt=query_prompt)

    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("What is your question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        asyncio.run(handle_query(prompt, query_chain, search, extract_urls, get_webpage_content))

    # サイドバーに情報を表示
    with st.sidebar:
        st.header("About")
        st.markdown("""
        このチャットボットは以下の機能を備えています：
        1. 最新情報が必要な場合はWeb検索を実行
        2. URLが含まれている場合はそのページの内容を解析
        3. 通常の会話にも対応
        """)

if __name__ == "__main__":
    main()