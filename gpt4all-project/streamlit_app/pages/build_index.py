import streamlit as st
import pandas as pd

from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain_community.document_loaders.directory import DirectoryLoader 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import WebBaseLoader 

from langchain_chroma.vectorstores import Chroma
from langchain_milvus.vectorstores import Milvus

from PIL import Image
import time
import os

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import utils.func 
import utils.constants as const

ETH_LOGO = Image.open('./images/logo.png')
ECP_LOGO = Image.open('./images/ecp_logo.png')
    
def on_settings_change():
    logging.info(" --- settings updated ---")

def on_local_model_change():
    st.session_state.embed_model = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={'allow_download': 'True'})
    logging.info(f" --- st.session_state.embed_model=GPT4AllEmbeddings(model_name='all-MiniLM-L6-v2.gguf2.f16.gguf', gpt4all_kwargs={'allow_download': 'True'}) ---")

def on_indexname_change():
    name = st.session_state.my_indexname
    vector_engine = st.session_state.vector_db
    name = utils.func.make_valid_directory_name(name)
    if os.path.exists(os.path.join(const.INDEX_ROOT_PATH, name)) and (vector_engine != 0):
        with container_name:
            st.error('The title name is not valid', icon="🚨")
    else:
        st.session_state.index_path_to_be_created = f"{const.INDEX_ROOT_PATH}/{name}"
        st.session_state.index_name = f"{name}"
        with container_name:
            if vector_engine == 0:
                st.markdown(f"Collection `{st.session_state.index_name}` will be created inside ChromaDB")
            if vector_engine == 1:
                st.markdown(f"`{st.session_state.index_path_to_be_created}.mvdb` will be created")

def on_docspath_change():
    logging.info("### on_docspath_change")
    dir = st.session_state.docspath
    with container_docs:
        with st.spinner('Checking files under the direcotry...'):
            files = utils.func.get_files_with_extensions(dir, const.SUPPORTED_FILE_TYPES)
            total_docs_size = utils.func.get_total_size_mib(dir)
            md = f"**`{len(files)}`** files found! (Total file size: **`{total_docs_size:,.2f}`** MiB)"
            logging.info(f"{len(files)} files found!")
            df = pd.DataFrame(files, columns=['Filename', 'Size (KiB)'])
            st.markdown(md)
            st.session_state.num_of_files_to_read = len(files)
            if len(files) != 0:
                st.dataframe(df.style.format({'Size (KiB)' : "{:,.1f}"}))

def on_urllist_change():
    urls = st.session_state.my_urllist
    if utils.func.check_urls(urls):
        st.session_state.num_of_urls_to_read = utils.func.count_urls(urls)
        with container_urls:
            st.success(f"{utils.func.count_urls(urls)} URLs supplied.", icon="✅")
        st.session_state.urllist = utils.func.extract_urllist(urls)
    else:
        st.session_state.num_of_urls_to_read = 0
        with container_urls:
            st.error("Invalid URL(s) contained.", icon="🚨")
        st.session_state.ready_to_index = False

def check_if_ready_to_index():
    logging.info("### check_if_ready_to_index()")
    if hasattr(st.session_state, "index_path_to_be_created"):
        is_name_ready = len(st.session_state.index_path_to_be_created)
    else:
        is_name_ready = False
    logging.info(f"is_name_ready: {is_name_ready}")
    if hasattr(st.session_state, "num_of_files_to_read"):
        num_of_files = st.session_state.num_of_files_to_read
    else:
        num_of_files = 0
    logging.info(f"num_of_files : {num_of_files}")
    if not hasattr(st.session_state, "num_of_urls_to_read"):
        st.session_state.num_of_urls_to_read = 0
    num_of_urls = st.session_state.num_of_urls_to_read
    logging.info(f"num_of_urls  : {num_of_urls}")
    if is_name_ready and (num_of_files or num_of_urls):
        logging.info("### check_if_ready_to_index() ---> Ready")
        st.session_state.index_button_disabled = False

def get_vector_engine_name():
    if st.session_state.vector_db == 0:
        return "ChromaDB"
    else:
        return "Milvus"

def create_index(docs):
    if(('chunk_size' not in st.session_state) or ('chunk_overlap' not in st.session_state)):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
    documents = text_splitter.split_documents(docs)

    if st.session_state.vector_db == 0:
        logging.info("### Creating ChromaDB Index...")
        vector_store = Chroma(
            embedding_function=st.session_state.embed_model,
            persist_directory="./chromadb",
            collection_name=st.session_state.index_name,
        )
        
        vector_store.add_documents(documents=documents)

        return vector_store
    if st.session_state.vector_db == 1:
        logging.info("### Creating Milvus Index...")
        vector_store = Milvus(
            embedding_function=st.session_state.embed_model,
            auto_id=True,
            connection_args={
                "uri": st.session_state.index_path_to_be_created+".db",
            },
        )
        vector_store.add_documents(documents=documents, embedding=st.session_state.embed_model)
        return vector_store


# App title
st.set_page_config(page_title="everyware copilot - Build Index", menu_items=None)
st.session_state.embed_model = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={'allow_download': 'True'})

### Building Index with Embedding Model
def index_data():
    with container_status:
        start_time = time.time()
        with st.status("Indexing documents..."):
            logging.info(f"Setting Embedding model... {st.session_state.embed_model}")
            logging.info(f"Setting Vecor DB Engine... {get_vector_engine_name()}")
            all_docs = []
            if st.session_state.num_of_files_to_read != 0:
                reader = DirectoryLoader(
                    path=st.session_state.docspath, 
                    recursive=True,
                    loader_cls=PyPDFLoader,
                    )
                st.write(    "Loading local documents...")
                logging.info("Loading local documents...")
                docs = reader.load()
                st.write(    f"{len(docs)} local documents loaded.")
                logging.info(f"{len(docs)} local documents loaded.")
                st.write(    "Building Index from local docs (using GPU)...")
                logging.info("Building Index from local docs (using GPU)...")

                all_docs.extend(docs)
            if st.session_state.num_of_urls_to_read != 0:
                st.write(    "Loading web documents...")
                logging.info("Loading web documents...")

                for url in st.session_state.urllist:
                    st.write(    f"Loading web documents from {url}...")
                    #scraper = SitemapLoader(
                    #    web_path=url,
                    #    max_depth=st.session_state.web_crawling_depth,
                    #)
                    #web_docs=scraper.load()
                    loader = WebBaseLoader(web_path = url)
                    web_docs = (loader.load())
                    st.write(    f"{len(web_docs)} web documents loaded from {url}.")
                    logging.info(f"{len(web_docs)} web documents loaded from {url}.")
                    logging.info(f"len(web_docs): {len(web_docs)}")
                    all_docs.extend(web_docs)

                #else:
                    #if st.session_state.vector_db == 0:
                    #    for d in web_docs:
                    #        index.add_documents(documents=d)
                    #else:
                    #    for d in web_docs:
                    #        index.upsert(documents=d)

            st.write(    "Building Index from web docs (using GPU)...")
            logging.info("Building Index from web docs (using GPU)...")
            index = create_index(all_docs)

            st.write(    "Saving the built index to disk...")
            logging.info("Saving the built index to disk...")
            st.write(    "Indexing done!")
            logging.info("Indexing done!")
        end_time = time.time()
        elapsed_time = end_time - start_time
    
    total_size_mib = utils.func.get_total_size_mib(st.session_state.index_path_to_be_created)

    md = f"""
    Index named **"{st.session_state.index_name}"** was built from `{len(all_docs)}` documents!

    The index is saved under `{st.session_state.index_path_to_be_created}` and the total size of this index is **`{total_size_mib:.2f}`** MiB. 

    The indexing task took **`{elapsed_time:.1f}`** seconds to competele.
    """

    with container_result:
        st.markdown(md)
        logging.info(md)

if "vector_db" not in st.session_state.keys():
    st.session_state.vector_db = 0

# Side bar
with st.sidebar:
    st.title("Building Index")
    st.logo(ETH_LOGO)
    st.image(ECP_LOGO, width=300)    
    st.info('Build your own custom Index based on your local/online documents.')

    st.subheader("Embedding Model")
    t1 = st.tabs(['Local'])[0]
    with t1:
        models = ["all-MiniLM-L6-v2.gguf2.f16.gguf"]
        st.selectbox("Predefined local embeddign model", models, index=models.index("all-MiniLM-L6-v2.gguf2.f16.gguf"), key='my_local_model', on_change=on_local_model_change)
    
    v_idx = st.radio("Choose your preferred Vector Database", ["ChromaDB", "Milvus"], index=st.session_state.vector_db)
    if v_idx == "ChromaDB":
        st.session_state.vector_db = 0
    elif v_idx == "Milvus":
        st.session_state.vector_db = 1

    use_customized_chunk = st.toggle("Customize chunk parameters", value=False)
    if use_customized_chunk:
        st.session_state.chunk_size = st.slider("Chunk size", 100, 2000, 512, key='my_chunk_size', on_change=on_settings_change)
        st.session_state.chunk_overlap = st.slider("Chunk overlap", 10, 400, 20, key='my_chunk_overlap', on_change=on_settings_change)
        logging.info(f"> Setting hunk_size    = {st.session_state.chunk_size}")
        logging.info(f"> Setting chunk_overlap = {st.session_state.chunk_overlap}")

    use_customized_web_crawler = st.toggle("Customize web crawling parameters", value=False)
    if use_customized_web_crawler:
        st.session_state.web_crawling_depth = st.slider("Depth", 0, 10, 2, key='my_crawling_depth')
    else:
        st.session_state.web_crawling_depth = 2

    st.page_link("app.py", label="Back to home", icon="🏠")

st.subheader("Index Name")
index_name = st.text_input("Enter the name for your new index", key='my_indexname', on_change=on_indexname_change)
container_name = st.container()

st.subheader('Local documents')
subdirs = utils.func.get_subdirectories(const.DOC_ROOT_PATH)
subdirs.insert(0, "")
st.selectbox("Select the path to the local directory used to store your documents", subdirs, key='docspath', on_change=on_docspath_change)
container_docs = st.container()
if len(subdirs) != 0:
    on_docspath_change()
else:
    st.session_state.num_of_files_to_read = 0

st.subheader('Online documents')
list_urls = st.text_area("List of URLs (one per line)", key='my_urllist', on_change=on_urllist_change)
container_urls = st.container()

st.warning("Check the model and its configurations on the sidebar (⬅️) and then hit the button below to build a new Index.", icon="⚠️")

container_settings = st.container()

check_if_ready_to_index()
logging.info(f"Setting Embedding model... {st.session_state.embed_model}")

st.button("Build Index", on_click=index_data, key='my_button', disabled=st.session_state.get("index_button_disabled", True))
container_status = st.container()
container_result = st.container()
