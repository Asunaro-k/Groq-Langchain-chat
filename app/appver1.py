import streamlit as st

#langchain
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import PromptTemplate

#webページの認識やファイル操作もろもろ
import requests
from bs4 import BeautifulSoup
import re
import os
import asyncio
import json
from PIL import Image
import io

# AI用
import torch
from transformers import pipeline

# 画像キャプションモデルの初期化
@st.cache_resource
def load_caption_model():
    device = 0 if torch.cuda.is_available() else -1
    caption_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=device,
        max_new_tokens = 100
    )
    return caption_model

# 画像をキャプション化する関数
def generate_image_caption(image_file):
    try:
        caption_model = st.session_state.image_captioner
        image = Image.open(image_file)
        captions = caption_model(image)
        caption = captions[0]['generated_text'] if captions else "画像の説明を生成できませんでした。"
        return caption
    except Exception as e:
        return f"画像キャプションの生成中にエラーが発生しました: {str(e)}"

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
        return f"Webページの取得中にエラーが発生しました: {str(e)}"

# Web検索関数の追加
async def perform_web_search(query_chain, search_wrapper, prompt):
    try:
        # 質問の分析
        analysis = await query_chain.ainvoke(prompt)
        content = analysis.content if hasattr(analysis, 'content') else str(analysis)
        
        needs_search = "NEEDS_SEARCH: true" in content
        has_url = "HAS_URL: true" in content

        # URLが含まれている場合の処理
        if has_url:
            urls = extract_urls(prompt)
            if urls:
                webpage_content = get_webpage_content(urls[0])
                return f"URLコンテンツ: {webpage_content}"

        # Web検索が必要な場合
        if needs_search:
            st.markdown("""DuckDuckGoで検索中...""")
            search_query = re.search(r'SEARCH_QUERY: (.*)', content)
            if search_query:
                search_results = search_wrapper.run(search_query.group(1))
                return f"検索結果: {search_results}"

        return None
    except Exception as e:
        return f"検索中にエラーが発生しました: {str(e)}"

# 設定ファイルを読み込みもしくは直書き
CONFIG_FILE = 'app_config.json'

def load_config():
    """Load configuration from a JSON file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'model': 'llama-3.1-70b-versatile',
            'temperature': 0.7,
            'system_prompt': 'あなたは親切で賢明な、多様な質問に答えられる優秀なアシスタントです。',
            'max_tokens': 4096,
            'vision_enabled': True,
            'search_enabled': True
        }
    
def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# 設定ページ
def settings_page():
    st.title("アプリケーション設定")
    
    config = load_config()
    
    # Model Selection
    st.header("モデル設定")
    model_options = [
        "llama-3.1-70b-versatile", 
        "mixtral-8x7b-32768", 
        "gemma-7b-it"
    ]
    selected_model = st.selectbox(
        "AIモデルを選択", 
        model_options, 
        index=model_options.index(config['model'])
    )
    
    # Temperature Slider
    temperature = st.slider(
        "Temperature (創造性の調整)", 
        min_value=0.0, 
        max_value=1.0, 
        value=config['temperature'], 
        step=0.1
    )
    
    # System Prompt
    st.header("システムプロンプト")
    system_prompt = st.text_area(
        "AIの基本的な振る舞いを定義", 
        value=config['system_prompt'], 
        height=200
    )
    
    # Advanced Settings
    st.header("詳細設定")
    max_tokens = st.number_input(
        "最大トークン数", 
        min_value=1024, 
        max_value=8192, 
        value=config['max_tokens']
    )
    
    vision_enabled = st.toggle(
        "画像キャプション機能を有効化", 
        value=config.get('vision_enabled', True)
    )
    
    # Preset Roles
    st.header("プリセット役割")
    preset_roles = {
        "デフォルト": "あなたは親切で賢明な、多様な質問に答えられる優秀なアシスタントです。",
        "人生相談": "あなたは共感力があり、深い洞察力を持つ人生相談のスペシャリストです。",
        "うるさい関西人": "あなたはやたらうるさい関西人です。",
        "胡散臭い中国人": "あなたは胡散臭い片言の日本語を話す中国人です。"
    }
    
    selected_preset = st.selectbox("プリセットの役割", list(preset_roles.keys()))

    # 検索機能の設定を追加
    st.header("検索設定")
    search_enabled = st.toggle(
        "Web検索機能を有効化", 
        value=config.get('search_enabled', True)
    )
    
    if st.button("設定を保存"):
        config.update({
            'model': selected_model,
            'temperature': temperature,
            'system_prompt': system_prompt or preset_roles[selected_preset],
            'max_tokens': max_tokens,
            'vision_enabled': vision_enabled,
            'search_enabled': search_enabled
        })
        
        save_config(config)
        st.success("設定が正常に保存されました！")
    
    return config
    

# チャットページ
def chat_page(config):
    st.title("Enhanced Multimodal Chat")

    # セッション状態の初期化
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # モデルとラッパーの初期化
    st.session_state.llm = ChatGroq(
        model_name=config['model'],
        temperature=config['temperature']
    )
    
	# 既存のメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
	# 画像キャプションモデルの初期化
    if 'image_captioner' not in st.session_state:
        st.session_state.image_captioner = load_caption_model()

    # DuckDuckGo検索の初期化
    search_wrapper = DuckDuckGoSearchAPIWrapper(
        backend="api", #'api': APIモード（通常使用するモード）。'html': HTMLモード（HTMLパーシングによる検索）。'lite': 軽量モード（低リソースモード）。
        max_results=5, #取得する検索結果の最大件数。
        region="jp-jp", #地域コード 'wt-wt'（全世界）
        safesearch="off", #セーフサーチモード
        source="text", #'text': テキスト検索。'news': ニュース検索。
        time="w" #'d': 過去1日。'w': 過去1週間。'm': 過去1か月。'y': 過去1年。
	)

    # 質問分析のためのチェーン
    query_prompt = PromptTemplate(template=QUERY_PROMPT, input_variables=["question"])
    query_chain = query_prompt | st.session_state.llm
    
	# 画像アップロード
    if config.get('vision_enabled', True):
        with st.sidebar:
            uploaded_image = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption='アップロードされた画像')
            caption = generate_image_caption(uploaded_image)
            
            st.session_state.messages.append({
                "role": "system", 
                "content": f"画像の説明: {caption}"
            })
            
            st.write("画像の説明:", caption)

    # チャット入力
    prompt = st.chat_input("メッセージを入力")
    
    if prompt:
        # メッセージ履歴に追加
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })

        # ユーザーメッセージの表示
        with st.chat_message("user"):
            st.markdown(prompt)

        # Web検索の実行（有効な場合）
        search_result = None
        if config.get('search_enabled', True):
            search_result = asyncio.run(perform_web_search(query_chain, search_wrapper, prompt))

        # 応答の生成
        with st.chat_message("assistant"):
            # メッセージの準備
            llm_messages = [SystemMessage(content=config['system_prompt'])]
            
            # 検索結果があれば追加
            if search_result:
                llm_messages.append(SystemMessage(content=search_result))
            
            # 会話履歴の追加
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    llm_messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'system':
                    llm_messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    llm_messages.append(AIMessage(content=msg['content']))

            # 応答の生成
            response = st.session_state.llm.invoke(llm_messages)
            st.markdown(response.content)

        # アシスタントの応答を履歴に追加
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.content
        })

def main():
	# Streamlitアプリの設定
	st.set_page_config(
		page_title="Enhanced Groq LangChain Chat",
		page_icon=":speech_balloon:",
		layout="wide"
	)

	page = "チャット"

	# サイドバーで切り替えボタン
	page = st.sidebar.radio("ナビゲーション", 
		["チャット", "設定"], 
		index=0
	)

	# Load configuration
	config = load_config()

	# Render appropriate page
	if page == "チャット":
		chat_page(config)
	else:
		config = settings_page()

if __name__ == "__main__":
	main()