import os
import streamlit as st 
import chromadb

from langchain_community.llms import GPT4All 
from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain.schema import SystemMessage, HumanMessage 
from langchain_chroma.vectorstores import Chroma 
from langchain_milvus.vectorstores import Milvus 
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts.prompt import PromptTemplate 

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

# App title
st.set_page_config(page_title="everyware copilot", menu_items=None)

AVATAR_AI   = Image.open('./images/ecp_agent.png')
AVATAR_USER = Image.open('./images/ecp_user.png')
ETH_LOGO = Image.open('./images/logo.png')
ECP_LOGO = Image.open('./images/ecp_logo.png')

SYSTEM_PROMPT=(
"""You are an expert Q&A assistant, working as an employee in Eurotech, answering questions on Eurotech products.
Never offend or attack or use bad words against Eurotech.
Always answer the query using the provided context and not prior knowledge.
Some rules to follow:
1. Provide specific answers and enumerate in bullet points when appropriate
2. Never directly reference the given context in your answer.
3. DO NOT start the response with the statement 'Based on the provided context' or something similar.
If you don't know the answer, just say that you don't know the answer, don't try to make up an answer.
Always answer the user question below based on provided context.
Context: {context}
Question: {question}
"""
)

def find_saved_indexes() -> List[vector_db_index]:
    chroma_collections = chroma_db.list_collections()
    milvus_collections = utils.func.list_files(const.INDEX_ROOT_PATH, ".db")
    result = []
    for coll in chroma_collections:
        result.append(vector_db_index(engine= 0, name= coll.name))
    for coll in milvus_collections:
        result.append(vector_db_index(engine= 1, name= coll))
    return result

def load_index(index: vector_db_index):
    use_gpu = os.environ.get("USE_GPU", "").lower() in ("yes", "true", "t", "1")
    device = "gpu" if use_gpu else "cpu"
    st.session_state.embed_model = GPT4AllEmbeddings(model_name='all-MiniLM-L6-v2.gguf2.f16.gguf', device=device, gpt4all_kwargs={'allow_download': 'True'})
    if(index.engine == 0):
        index = Chroma(
            client=chroma_db,
            collection_name=index.name,
            embedding_function=st.session_state.embed_model,
        )
        return index
    if(index.engine == 1):
        index = Milvus(
            st.session_state.embed_model,
            connection_args={"uri": os.path.join(const.INDEX_ROOT_PATH, index.name+".mvdb")},
        )
        return index
    return None

def format_model_name(model):
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
        #st.session_state.chat_engine.reset()

def index_name_format(value: vector_db_index):
    if value.engine == 0:
        return "ChromaDB - "+value.name
    else:
        return "Milvus - "+value.name

if "old_selected_index" not in st.session_state.keys():
    st.session_state.old_selected_index = ''

# Side bar
with st.sidebar:
    st.logo(ETH_LOGO)
    st.image(ECP_LOGO, width=300)

    st.subheader('Your local AI assistant')
    st.html(body="<hr style='border: none; height: 4px; background-color: #410099;'\>")

    engine = st.radio("LLM engine:", ["GPT4All"])

    if engine == "GPT4All":
        models_dir = os.path.join("./gpt4all_models")
        models = os.listdir(models_dir)
        check_list = [model for model in models]
        if not check_list:
            with st.spinner('Downloaing Nous-Hermes 2 model ...'):
                GPT4All(model=os.path.join(models_dir, 'Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf'), allow_download=True)
                logging.info(" ### Downloaing Nous-Hermes 2 completed.")
        st.session_state["model"] = st.selectbox("Choose your LLM", models, index=models.index(check_list[0]))
        logging.info(f"> GPT4All model = {st.session_state.model}")
        st.page_link("pages/download_model.py", label=" Download a new LLM", icon="➕")
        
        use_gpu = os.environ.get("USE_GPU", "").lower() in ("yes", "true", "t", "1")
        device = "gpu" if use_gpu else "cpu"
        logging.info(f"> currently using: {device}")
        num_threads = os.cpu_count()
        
        llm = GPT4All(
            model=os.path.join(models_dir, st.session_state.model), 
            device=device,
            n_predict=512,
            n_threads=num_threads,
            allow_download=False,
        )    
               

    use_index = st.toggle("Use RAG", value=True)
    if use_index:
        saved_index_list = find_saved_indexes()
        index = next((i for i, item in enumerate(saved_index_list) if item.name.startswith('_')), None)
        st.session_state.selected_index = st.selectbox("Index", saved_index_list, index, format_func=index_name_format)
        reload_index() 
        logging.info(f"> selected_index = {st.session_state.selected_index}")
        st.page_link("pages/build_index.py", label=" Build a new index", icon="➕")

        st.session_state.top_k = 2
        use_custom_params = st.toggle("Customize retrieval options", value=False)
        if use_custom_params:
            st.session_state.top_k = st.slider("top k", 1, 10, 2, key='my_top_k_size')
            logging.info(f"> Top_K    = {st.session_state.top_k}")

        if st.session_state.selected_index != None and st.session_state.index != None:
        
            prompt = PromptTemplate.from_template(template=SYSTEM_PROMPT)

            st.session_state.llm_chain = (
                {
                    "context": RunnablePassthrough(),
                    "question": RunnablePassthrough(),
                }
                | prompt
                | llm
                | StrOutputParser()
            )

# initialize history
if "index" not in st.session_state.keys():
    st.session_state.index = None
if "messages" not in st.session_state.keys():
    clear_history()
if "old_selected_index" not in st.session_state.keys():
    st.session_state.old_selected_index = ''

def model_res_generator(question=""):
    start = time.time()
    if use_index:
        if st.session_state.selected_index == None:
            st.warning('No index selected!', icon="⚠️")
            return
        logging.info(f">>> RAG enabled:")

        context = st.session_state.index.similarity_search(query=question, k=st.session_state.top_k)
        for doc in context:
            if len(doc.page_content) > 512:
                doc.page_content = doc.page_content[:512] + "..."
        
        response = st.session_state.llm_chain.invoke({"context": context, "question": question})
        yield(response)
        end = time.time()
        logging.info(f"Time to generate the response: {end-start}")
    else:
        logging.info(f">>> Just LLM (no RAG):")
        messages_only_role_and_content = []
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                messages_only_role_and_content.append(SystemMessage(content=message["content"]))
            else:
                messages_only_role_and_content.append(HumanMessage(content=message["content"]))
        
        chat_response = llm.invoke( 
            messages_only_role_and_content,
        )
        yield chat_response
        end = time.time()
        logging.info(f"Time to generate the response: {end-start}")


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
            message = st.write(model_res_generator(prompt))
            st.button("clear conversation context", on_click=clear_history)
            st.session_state.messages.append({"role": "assistant", "content": message, "avatar": AVATAR_AI})
