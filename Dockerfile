FROM dustynv/python:r35.2.1

RUN git clone https://github.com/eurotech/everyware-copilot.git \
    && cd everyware-copilot \
    && pip3 install docker_py \
    && pip3 install ollama \
    && pip3 install streamlit \
    && pip3 install llama_index \
    && pip3 install llama_index.llms.ollama \
    && pip3 install llama_index.llms.nvidia \
    && pip3 install llama_index.embeddings.huggingface \
    && pip3 install chromadb \
    && pip3 install llama_index.vector_stores.chroma \
    && pip3 install llama_index.vector_stores.milvus \
    && pip3 install --force-reinstall torch==1.13.0

RUN cd /usr/local/bin/ \
    && wget https://www.sqlite.org/2024/sqlite-autoconf-3460100.tar.gz \
    && tar -zxvf sqlite-autoconf-3460100.tar.gz \
    && cd sqlite-autoconf-3460100 \
    && ./configure \
    && make \
    && make install \
    && cp /usr/local/lib/libsqlite3* /usr/local/cuda/lib64
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

WORKDIR /everyware-copilot/streamlit_app

ENTRYPOINT [ "streamlit", "run", "./app.py" ]
