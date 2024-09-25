import streamlit as st 
import pandas as pd

import time
import os
import requests
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

url = "https://raw.githubusercontent.com/nomic-ai/gpt4all/main/gpt4all-chat/metadata/models3.json"

grandparent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
models_dir = os.path.join(grandparent_dir, 'gpt4all_models')
models = os.listdir(models_dir)

# App title
st.set_page_config(page_title="Jetson Copilot - Download Model", menu_items=None)

st.subheader("List of Models Already Downloaded")
with st.spinner('Checking existing models hosted on GPT4All...'):
    response = requests.get(url)
    data = json.loads(response.text)

    models_data = []
    for model_name in models:
        for model_data in data:
            if model_data["filename"] == model_name:
                model_info = model_data
                models_data.append((
                    model_info['name'],
                    int(model_info['filesize']) / 1024 / 1024,
                    model_info['ramrequired'],
                    model_info['type'],
                    model_info['parameters'],
                    model_info['quant']
                ))
                break 

    logging.info(f"{len(models)} models found!")
    df = pd.DataFrame(models_data, columns=[
        'Name', 'Size(MiB)', 'Required RAM', 'Family', 'Parameters', 'Quantization'
    ])
    if len(models) != 0:
        st.dataframe(df.style.format({'Size(MiB)' : "{:,.1f}"}))

def on_newmodel_name_change():
    logging.info("on_newmodel_name_change()")
    newmodel_name = st.session_state.my_newmodel_name
    if newmodel_name.strip():
        logging.info("Name supplied")
        st.session_state.download_model_disabled = False
    else:
        logging.info("Name NOT supplied")
        st.session_state.download_model_disabled = True

def download_model():
    logging.info("download_model()")
    newmodel_name = st.session_state.my_newmodel_name
    with container_status:
        start_time = time.time() 
        my_bar = st.progress(0, text="progress text")

        response = requests.get(url)
        data = json.loads(response.text)

        try:
            for model_data in data:
                if newmodel_name.lower() in model_data["filename"].lower():
                    model_name = model_data["filename"]
                    model_url = model_data["url"]
                    for model in models:
                        if model == model_name:
                            raise ValueError(f"The model '{model_name}' is already present.")
                    break    
            response = requests.get(model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            file_path = os.path.join(models_dir, model_name)

            with open(file_path, 'wb') as file:
                for file_data in response.iter_content(chunk_size=1024):
                    file.write(file_data)
                    downloaded_size += len(file_data)
                    progress = int(downloaded_size / total_size * 100)
                    my_bar.progress(progress, text=f"Downloading... {progress}%")

            end_time = time.time()
            st.success(f"Download completed in {end_time - start_time:.2f} seconds")
        except ValueError as ve:
            logging.error(ve)
            st.error(f"The model '**`{model_name}`**' is already present.", icon="⚠️")
        except Exception as e:
            logging.error(f"A ResponseError occurred: {e}")
            st.error(f"It looks like \"**`{newmodel_name}`**\" is not the right name.", icon="🚨")
        

st.subheader("Download a New Model")
st.info(f"Check the model name on [GPT4All models]({url}) page.", icon=":material/info:")

model_name = st.text_input(
    "Name of model to download",
    key='my_newmodel_name', 
    on_change=on_newmodel_name_change
)
st.button(
    "Download Model", 
    key='my_button', 
    on_click=download_model, 
    disabled=st.session_state.get("download_model_disabled", True)
)
container_status = st.container()

st.page_link("app.py", label="Back to home", icon="🏠")