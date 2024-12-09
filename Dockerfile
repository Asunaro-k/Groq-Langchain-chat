#FROM langchain/langchain
#FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM python:3.11 
#など公式のコンテナ（環境設定）の読み込み

# Pythonのシンボリックリンクを作成（pythonコマンドを使用可能にする）
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# 依存関係のインストール
COPY requirements.txt .
RUN pip install -r requirements.txt

# ポートの指定
EXPOSE 8501
# Streamlitの起動
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501","--server.address=0.0.0.0" ]