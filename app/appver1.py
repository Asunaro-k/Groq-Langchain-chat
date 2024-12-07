import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import re
import os
import asyncio
import json
from PIL import Image
import io

# 追加の必要なインポート
import torch
from transformers import pipeline

# グローバル変数としてキャプションモデルを初期化
@st.cache_resource
def load_caption_model():
    # 利用可能なデバイスを動的に判定
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
        # キャプションモデルの取得
        caption_model = st.session_state.image_captioner
        
        # 画像をPILで開く
        image = Image.open(image_file)
        
        # キャプション生成
        captions = caption_model(image)
        
        # キャプションの取得（通常は最初の結果を使用）
        caption = captions[0]['generated_text'] if captions else "画像の説明を生成できませんでした。"
        
        return caption
    except Exception as e:
        return f"画像キャプションの生成中にエラーが発生しました: {str(e)}"

# Configuration Management
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
            'vision_enabled': True
        }

def save_config(config):
    """Save configuration to a JSON file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# URL Detection Function
def extract_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

# Settings Page
def settings_page():
    st.title("アプリケーション設定")
    
    # Load current configuration
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
        "プログラミング講師": "あなたは経験豊富なプログラミング講師です。",
        "創作アシスタント": "あなたは創造性豊かな作家兼アイデアジェネレーターです。"
    }
    
    selected_preset = st.selectbox("プリセットの役割", list(preset_roles.keys()))
    
    if st.button("設定を保存"):
        # Update configuration
        config.update({
            'model': selected_model,
            'temperature': temperature,
            'system_prompt': system_prompt or preset_roles[selected_preset],
            'max_tokens': max_tokens,
            'vision_enabled': vision_enabled
        })
        
        save_config(config)
        st.success("設定が正常に保存されました！")
    
    return config

# Main Chat Application
def chat_page(config):
    st.title("Enhanced Multimodal Chat")

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # LLM Configuration
    st.session_state.llm = ChatGroq(
        model_name=config['model'],
        temperature=config['temperature']
    )

    # 画像キャプションモデルの初期化
    if 'image_captioner' not in st.session_state:
        st.session_state.image_captioner = load_caption_model()

    # Image Upload
    if config.get('vision_enabled', True):
        uploaded_image = st.file_uploader("画像をアップロード", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_image is not None:
            # Display the uploaded image
            st.image(uploaded_image, caption='アップロードされた画像')
            
            # Generate image caption
            caption = generate_image_caption(uploaded_image)
            
            # Add image caption to messages
            st.session_state.messages.append({
                "role": "system", 
                "content": f"画像キャプション: {caption}"
            })
            
            # Display the generated caption
            st.write("画像キャプション:", caption)

    # Chat input
    prompt = st.chat_input("メッセージを入力")
    
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            # Prepare messages for the LLM
            llm_messages = [SystemMessage(content=config['system_prompt'])]
            for msg in st.session_state.messages:
                if msg['role'] == 'user':
                    llm_messages.append(HumanMessage(content=msg['content']))
                elif msg['role'] == 'system':
                    llm_messages.append(SystemMessage(content=msg['content']))
                elif msg['role'] == 'assistant':
                    llm_messages.append(AIMessage(content=msg['content']))

            # Generate response
            response = st.session_state.llm.invoke(llm_messages)
            st.markdown(response.content)

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response.content
        })

def main():
    # Streamlit page configuration
    st.set_page_config(
        page_title="Enhanced Multimodal Chat", 
        page_icon=":robot:", 
        layout="wide"
    )

    # Create a sidebar for navigation
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