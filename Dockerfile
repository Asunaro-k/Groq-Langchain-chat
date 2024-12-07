#FROM langchain/langchain
#FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
FROM python:3.11 

#ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

#RUN apt-get update && apt-get install -y libffi-dev libnacl-dev
# RUN apt-get update && apt-get install -y libffi-dev libnacl-dev 
    #libnvrtc12 \
    #cuda-toolkit-11-8 \
# RUN apt-get update && apt-get install -y \
#     libffi-dev \
#     libnacl-dev \
#     python3 \
#     python3-pip \
#     nvidia-cuda-dev \
#     cuda-nvrtc-12-1 \
#     cuda-nvrtc-dev-12-1 \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# Pythonのシンボリックリンクを作成（pythonコマンドを使用可能にする）
RUN ln -s /usr/bin/python3 /usr/bin/python

# RUN apt-get install -y python3-dev
# Set the working directory in the container
WORKDIR /app
# Dockerイメージにsd_embedディレクトリをコピー

#RUN apt-get update && apt-get install -y cmake build-essential libboost-dev
#RUN pip install --upgrade setuptools wheel
#RUN pip install pyarrow --upgrade --prefer-binary
RUN pip install streamlit
RUN python3 -m pip install beautifulsoup4
RUN python3 -m pip install langgraph langsmith
RUN python3 -m pip install langchain-groq
RUN python3 -m pip install Pillow
RUN python3 -m pip install aiohttp
RUN python3 -m pip install langchain_google_community
RUN python3 -m pip install -U duckduckgo-search
RUN python3 -m pip install torch==2.1.0
RUN python3 -m pip install torchvision==0.16.0
RUN python3 -m pip install torchaudio==2.1.0
RUN python3 -m pip install transformers==4.46.2
# ポートの指定
EXPOSE 8501
# Streamlitの起動
ENTRYPOINT ["streamlit", "run", "appver1.py", "--server.port=8501","--server.address=0.0.0.0" ]