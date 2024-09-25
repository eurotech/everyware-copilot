# everyware copilot

everyware copilot is a reference application for a ***local*** AI assistant.

![everyware copilot architecture](./ECP.png)

It demonstrates two things;

- Running open-source LLMs (large language model) on device
- Augumenting the LLM to have access to your locally indexed knowledge (**RAG**, retrieval-augmented generation)

> [!IMPORTANT]
>
> This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
>
> | OSS | Repo URL | Usage |
> | --- | --- | ----- |
> | [Ollama](https://www.ollama.com/) | [GitHub](https://github.com/ollama/ollama) | To host and run LLMs locally, including embedding models for building index from documents |
> | [LlamaIndex](https://www.llamaindex.ai/) | [GitHub](https://github.com/run-llama/llama_index) | Data framework for LLM, used mainly to realize RAG pipeline. |
> | [Streamlit](https://streamlit.io/) | [GitHub](https://github.com/streamlit/streamlit) | Python library to create an interactive web app |
> | [Hugging Face](https://huggingface.co/) | [Hugging Face](https://huggingface.co/WhereIsAI/UAE-Large-V1) | Embedding Model used to create the indexes |
> | [NVIDIA IoT](https://github.com/NVIDIA-AI-IOT) | [GitHub](https://github.com/NVIDIA-AI-IOT/jetson-copilot) | The parent repository of everyware copilot |

## 🏃 Getting started

### First time setup

If this is your first time running everyware copilot, first run `setup_environment.sh` to ensure you have all the necessary software installed and the environment set up.

```bash
git clone https://github.com/eurotech/everyware-copilot/
cd everyware-copilot
./setup_environment.sh
```

It will install the following, if not yet available.

- Chromium web browser
- Docker

### How to start everyware copilot

```bash
cd everyware-copilot
./launch_everyware_copilot.sh
```

This will start a Docker container and start a Ollama server and Streamlit app inside the container. It will show the URL on the console in order to access the web app hosted on your hardware.

With your web browser, open the **Local URL** (`localhost`). Or on a PC connected to the same network as on your hardware, access the **Network URL**.

```bash
Local URL: http://localhost:8501
Network URL: http://10.110.50.252:8501 
```

> [!NOTE]
> You will need an active Internet connection when everyware copilot launches for the first time, as it will pull the needed container images (and download the default LLM and embedding model when web UI starts for the first time).

When you access the web UI for the first time, it will dowload the default LLM (`llama3`) and the embedding model (`mxbai-embed-large`).

> [!TIP]
> If you are on Ubuntu Desktop, a frameless Chromium window will pop up to access the web app, to make it look like an independent application.
> You need to close the window as stopping the container on the console won't shutdown Chromium.
>

## 📖 How to use everyware copilot

### 0. Interact with the plain Llama3 (8b)

You can use everyware copilot just to interact with a LLM withut enabling Retrieval-Augmented Generation.

By default, Llama3 (8b) model is downloaded when running for the first time and used as the default LLM.

You will be surprised by how much a model like Llama3 is capable of, but may soon find limitations as it does not have information prior to its cutoff date nor know anything about your specific subject matter.

### 1. Ask the copilot questions using the pre-built index

On the side panel, you can toggle "Use RAG" on to enable RAG pipeline.\
The LLM will have an access to a custom knowledge/index that is selected under "Index".

As a sample, a pre-build index "`EUROTECH_DSS`" is provided.\
This is built on several Eurotech Product DataSheets.

> It is mounted as `/media/<USER_NAME>/EUROTECH_DSS/` once you execute `udisksctl mount -b /dev/disk/by-label/EUROTECH_DSS`.

You can ask questions like:

``` shell
What are the differences between a ReliaGATE 10-14 and a ReliaGATE 10-12?
```

### 2. Build your own index based on your documents

You can build your own index based on your local and/or online documents.

First, on the console (or on the desktop) create a directory under `Documents` directory to store your documents.

```bash
cd eurotech-copilot
mkdir Documents/My-Knowledge-Base
cd Documents/My-Knowledge-Base
wget https://my_document_registry/docs/my-documentation.pdf
```

Now back on the web UI, open the side bar, toggle on "Use RAG", then click on "**➕Build a new index**" to jump to a "**Build Index**" page.\
Choose your preferred Vector Store technology and customize embedding parameters.

Give a name for the Index you want to build. (e.g. "My Knowledge Base")\
Fill in the field and hit `Enter`. The copilot will show what path will be created for your index, basing on the vector store of your choice.

> ![alt text](index_name_checked.png)

From the drop down box under "**Local documents**", select the directory you created and saved your documents in. (e.g. `/opt/eurotech_copilot/Documents/My-Knowledge-Base`).

It will show the summary of files found in the selected directory.

> ![alt text](local_documents_selected.png)

If you want to supply additional URLs for online documents to be ingested, fill the text area with one URL per a line.

The copilot will crawl the provided URLs recursively (recursion depth can be configured in the side panel) using Chromium/Chrome as a crawling engine.\
You can skip this if you are building your index only based on your local documents.

> [!NOTE]
> Current Embedding Model is `WhereIsAI/UAE-Large-V1` from [Hugging Face](https://huggingface.co/WhereIsAI/UAE-Large-V1)\
> Use of OpenAI embedding models is not well supported and needs more testing.

Finally, hit "**Build Index**" button.\
It will show the progress in the drop-down "status container", so you can check the status by clicking on it.\
Once done, it will show the summary of your index and time it took.

You can go back to the home screen to now select the index you just built.

### 3. Test different LLM or Embedding model

TODO

## 🏗️ Development

Streamlit based web app is very easy to develop.

On web UI, at the top-right of the screen, choose "**Always rerun**" to automatically update your app every time you change the source codes.

See [Streamlit Documentation](https://docs.streamlit.io/get-started/fundamentals/main-concepts#development-flow) for the detail.

### Manually run streamlit app inside the container

In case you make more fundamental changes, you can also manually run streamlit app.

```bash
cd everyware-copilot
./launch_dev.sh
```

Once in container;

```bash
streamlit run app.py
```

## 🧱 Directory structure

``` bash
└── everyware-copilot
    ├── launch_everyware_copilot.sh
    ├── setup_environment.sh
    ├── Documents 
    │   └── your_abc_docs
    ├── Indexes
    │   ├── EUROTECH_DSS
    │   └── your_abc_index
    ├── logs
    │   ├── container.log
    │   └── ollama.log
    ├── ollama_models
    └── Streamlit_app
        ├── pages
        │   ├── build_index.py
        │   └── download_model.py
        └── app.py
```

Following directories inside the `everyware-copilot` directory are mounted in the Docker container.

| Directory Name | Description |
| -------------- | ----------- |
| `Docuemtns`    | Directory to store your documents to be indexed |
| `Indexes`      | Directory to store pre-built (or built-by-you) indexes for LLM to perform RAG on |
| `logs`         | Directory for the app to store log files |
| `ollama_models`| Directory for the ollama server to store download models |
| `stremlit_app` | Directory for Python scripts to make up the web app |

## 💫 Troubleshooting

If you find any issue, please check [GitHub Issues of the everyware copilot repo](https://github.com/eurotech/everyware-copilot/issues).

## Supposed usage

You can use everyware copilot in multiple ways.

### Useful AI tool

Users can easily run an LLM on bare metal hardware (including Jetson) without relying on any cloud services.\
They may find the AI assistantce on some tasks useful, like to find out the right  command to use on Linux system. They can even expand the LLM knowledge by building the local index based on their own documents that LLM can access.

### Reference for budling a custom AI assistant

Developers can use everyware copilot as a reference for building their own AI assistant on specific domain area's or product's knowledge.

## ⚖️ License

Please see LICENSE file.

## 📜 Project status

Pushed to public, still partially in development.

### TODO

- [ ] Update videos
- [ ] README revamp
- [ ] Support OpenAI embedding models
- [ ] Download and launch NVIDIA Inference Microservices
- [ ] Implement request queue to fully decouple frontend and backend
