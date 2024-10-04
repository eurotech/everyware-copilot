import os
import ollama
# import openai
import streamlit as st
import docker
import chromadb

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import load_index_from_storage, StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore

from PIL import Image
from typing import List
import time

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

import utils.func
from utils.func import vector_db_index 
import utils.constants as const

chroma_db = chromadb.PersistentClient(path="./chromadb")

# As we cannot retrieve individual collections from Milvus, we have to use a
# new Milvus DB for each individual index
# milvus_db = MilvusClient("./milvus_db.db")

# App title
st.set_page_config(page_title="everyware copilot", menu_items=None)

AVATAR_AI   = Image.open('./images/ecp_agent.png')
AVATAR_USER = Image.open('./images/ecp_user.png')
ETH_LOGO = Image.open('./images/logo.png')
ECP_LOGO = Image.open('./images/ecp_logo.png')

SYSTEM_PROMPT=(
"""You are an expert Q&A assistant, working as an employee in Eurotech, answering questions on Eurotech products.
Never offend or attack or use bad words against Eurotech.
Always answer the query using the provided context from the Eurotech documentation and not prior knowledge.
Some rules to follow:
1. Provide specific answers in bullet points if you are returing a list or a sequence of steps
2. Never directly reference the given context in your answer.
3. DO NOT start the response with the statement 'According to the provided documentation' or 'Based on the provided documents' or 'Based on the provided documentation' or something similar.
"""
)

CONTEXT_PROMPT=(
"""Here are the relevant documents for the context:
---------------------
{context_str}
---------------------
Based on the above documents, provide a detailed answer for the user question below.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
DO NOT start the response with the statement 'According to the provided documentation' or 'Based on the provided documents' or 'Based on the provided documentation' something similar.
"""
)

def find_saved_indexes() -> List[vector_db_index]:
    json_collections = utils.func.list_directories(const.INDEX_ROOT_PATH)    
    chroma_collections = chroma_db.list_collections()
    milvus_collections = utils.func.list_files(const.INDEX_ROOT_PATH, ".mvdb")
    print(milvus_collections)
    result = []
    for coll in json_collections:
        result.append(vector_db_index(engine= 0, name= coll))
    for coll in chroma_collections:        
        result.append(vector_db_index(engine= 1, name= coll.name))
    for coll in milvus_collections:
        result.append(vector_db_index(engine= 2, name= coll))
    return result

def load_index(index: vector_db_index):
    Settings.embed_model = HuggingFaceEmbedding(model_name="WhereIsAI/UAE-Large-V1", 
                                            trust_remote_code=True) #TODO
    if(index.engine == 0):
        # JSON
        dir = f"{const.INDEX_ROOT_PATH}/{index.name}"
        storage_context = StorageContext.from_defaults(persist_dir=dir)
        index = load_index_from_storage(storage_context)
        return index
    if(index.engine == 1):
        # ChromaDB
        chroma_collection = chroma_db.get_or_create_collection(index.name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store
        )
        return index
    if(index.engine == 2):
        vector_store = MilvusVectorStore(os.path.join(const.INDEX_ROOT_PATH, index.name+".mvdb"), dim=1024)
        index = VectorStoreIndex.from_vector_store(
            vector_store
        )
        return index
    return None

def format_model_name(model):
    print(model)
    return model[0]

def reload_index():    
    logging.info(f"> selected_index = {st.session_state.selected_index}")
    logging.info(f"> old_selected_index = {st.session_state.old_selected_index}")
    if st.session_state.old_selected_index != st.session_state.selected_index:
        st.session_state.old_selected_index = st.session_state.selected_index
        logging.info(f"> replaced old_selected_index = {st.session_state.old_selected_index}")
        if st.session_state.selected_index != None:
            with st.spinner('Loading Index...'):
                st.session_state.index = load_index(st.session_state.selected_index)
                logging.info(f" ### Loading Index '{st.session_state.selected_index}' completed.")

def clear_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me any question about Eurotech!", "avatar": AVATAR_AI}
    ]
    if st.session_state.index != None:
        logging.info("> clearing chat context")
        st.session_state.chat_engine.reset()

def index_name_format(value: vector_db_index):
    if value.engine == 1:
        return "ChromaDB - "+value.name
    if value.engine == 2:
        return "Milvus - "+value.name
    return "JSON - "+value.name

# Side bar
with st.sidebar:        
    # # Add css to make text smaller
    # st.markdown(
    #     """<style>textarea { font-size: 0.8rem !important; } </style>""",
    #     unsafe_allow_html=True,
    # )
    st.logo(ETH_LOGO)
    # st.image(ETH_LOGO, use_column_width=True)    
    # st.title(":airplane: Eurotech Copilot")
    st.image(ECP_LOGO, width=300)    

    st.subheader('Your local AI assistant')
    st.html(body="<hr style='border: none; height: 4px; background-color: #410099;'\>")

    # models = [model["name"] for model in ollama.list()["models"]]
    dockerClient = docker.from_env()

    # t1,t2 = st.tabs(['Ollama','NVIDIA NIM'])
    engine = st.radio("Choose your preferred LLM Engine", ["Ollama","NVIDIA NIM"])

    if engine == "Ollama":
        models = [[model["name"], ""] for model in ollama.list()["models"]]
        check_list = [model[0] for model in models]
        if 'llama3.1:latest' not in check_list:
            with st.spinner('Downloaing llama3.1 model ...'):
                ollama.pull('llama3.1')
                logging.info(" ### Downloaing llama3.1 completed.")

        if 'mxbai-embed-large:latest' not in check_list:
            with st.spinner('Downloaing mxbai-embed-large model ...'):
                ollama.pull('mxbai-embed-large')
                logging.info(" ### Downloaing mxbai-embed-large completed.")
        st.session_state["model"] = st.selectbox("Choose your LLM", models, format_func=format_model_name, index=models.index(['llama3.1:latest', '']))
        logging.info(f"> Ollama model = {st.session_state.model}")
        st.page_link("pages/download_model.py", label=" Download a new LLM", icon="➕")
        Settings.llm = Ollama(model=st.session_state["model"][0], request_timeout=300.0)        
    else:
        print(dockerClient.containers.list())
        models = [[container.labels["com.nvidia.nim.model"], container.labels["llm_port"]] for container in dockerClient.containers.list()]
        st.session_state["model"] = st.selectbox("Choose your LLM", models, format_func=format_model_name, index=0)
        logging.info(f"> NVIDIA NIM model = {st.session_state.model}")
        img_name = st.session_state["model"][0]
        img_port = st.session_state["model"][1]
        Settings.llm = NVIDIA(model=img_name, base_url=f"http://127.0.0.1:{img_port}/v1")

    use_index = st.toggle("Use RAG", value=False)
    if use_index:
        saved_index_list = find_saved_indexes()
        index = next((i for i, item in enumerate(saved_index_list) if item.name.startswith('_')), None)
        st.session_state.selected_index = st.selectbox("Index", saved_index_list, index, format_func=index_name_format)
        reload_index()
        logging.info(f"> selected_index = {st.session_state.selected_index}")
        
        st.page_link("pages/build_index.py", label=" Build a new index", icon="➕")

        top_k = 40
        top_n = 3
        use_custom_params = st.toggle("Customize retrieval options", value=False)
        if use_custom_params:
            top_k = st.slider("top k", 4, 100, 40, key='my_top_k_size')
            top_n = st.slider("top n", 1, 10, 3, key='my_top_n_size')
            logging.info(f"> Top_K    = {top_k}")
            logging.info(f"> Top_N = {top_n}")

        if st.session_state.selected_index != None:
            system_prompt = st.text_area("System prompt", SYSTEM_PROMPT, height=240)
            
            context_prompt = st.text_area("Context prompt", CONTEXT_PROMPT, height=240)            
            logging.info(f"> system_prompt = {system_prompt}")
            logging.info(f"> context_prompt = {context_prompt}")

            # init models
            if st.session_state.index != None:
                retriever = st.session_state.index.as_retriever(similarity_top_k=top_k)

                st.session_state.chat_engine = ContextChatEngine.from_defaults(
                    retriever= retriever,
                    system_prompt= system_prompt,    
                    context_template= context_prompt,
                    node_postprocessors=[SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-4-v2", top_n=top_n)],                                                            
                )

                # st.session_state.chat_engine = st.session_state.index.as_chat_engine(
                #     chat_mode="context", 
                #     streaming=True,
                #     memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
                #     llm=Settings.llm,
                #     context_prompt=(context_prompt),
                #     verbose=True)

# initialize history
if "index" not in st.session_state.keys():
    st.session_state.index = None
if "messages" not in st.session_state.keys():
    clear_history()
if "old_index_name" not in st.session_state.keys():
    st.session_state.old_selected_index = ''

def model_res_generator(prompt=""):
    if use_index:
        if st.session_state.selected_index == None:
            st.warning('No index selected!', icon="⚠️")
            return            
        logging.info(f">>> RAG enabled:")
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        for chunk in response_stream.response_gen:
            yield chunk
    else:
        logging.info(f">>> Just LLM (no RAG):")
        messages_only_role_and_content = []
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                messages_only_role_and_content.append(ChatMessage(role=MessageRole.SYSTEM, content=(message["content"])))
            else:
                messages_only_role_and_content.append(ChatMessage(role=MessageRole.USER, content=(message["content"])))

        chatResponse = Settings.llm.chat(            
            messages=messages_only_role_and_content,
        )
        yield chatResponse.message.content


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=message["avatar"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Enter prompt here..."):    
    # add latest message to history in format {role, content}
    st.session_state.messages.append({"role": "user", "content": prompt, "avatar": AVATAR_USER})

    with st.chat_message("user", avatar=AVATAR_USER):
        st.markdown(prompt)

# if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant", avatar=AVATAR_AI):
        with st.spinner("Thinking..."):
            time.sleep(1)
            message = st.write_stream(model_res_generator(prompt))
            st.button("clear conversation context", on_click=clear_history)
            st.session_state.messages.append({"role": "assistant", "content": message, "avatar": AVATAR_AI})
