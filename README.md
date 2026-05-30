# RAG-Based-Medical-Chatbot

A Retrieval-Augmented Generation (RAG) powered medical chatbot that answers health-related questions by retrieving relevant information from medical PDF documents and generating context-aware responses using a large language model.

## Overview

This project combines the power of vector search and LLMs to build an intelligent medical Q&A system. Medical documents (PDFs) are processed, chunked, and stored as vector embeddings in a Pinecone vector database. When a user asks a question, the most relevant document chunks are retrieved and passed to the LLM (Llama 3.1 via Groq) to generate a grounded, accurate response — minimizing hallucinations by anchoring answers in real source material.

## Tech Stack

| Component | Technology |
|---|---|
| Web Framework | Flask |
| LLM | Llama 3.1 8B Instant (via Groq API) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers) |
| Vector Database | Pinecone (Serverless, AWS us-east-1) |
| Orchestration | LangChain |
| PDF Parsing | PyPDF |
| Environment Config | python-dotenv |

## Project Structure
RAG-Based-Medical-Chatbot/
│
├── data/                   # Medical PDF documents (knowledge base)
├── research/               # Jupyter notebooks for experimentation
│   └── trials.ipynb
├── src/                    # Core source package
│   ├── init.py
│   ├── helper.py           # PDF loading, chunking, embedding utilities
│   └── prompt.py           # System prompt definition for the LLM
├── static/                 # Static assets (CSS, JS) for the web UI
├── templates/              # HTML templates (chat.html)
│   └── chat.html
├── app.py                  # Flask application entry point
├── store_index.py          # One-time script to build and store vector index
├── setup.py                # Package setup (author: Ratul Podder)
├── requirements.txt        # Python dependencies
├── template.sh             # Shell script to scaffold project directory structure
├── .gitignore
└── LICENSE                 # Apache-2.0
## How It Works

1. **Data Ingestion** (`store_index.py`): Medical PDFs from the `data/` directory are loaded and parsed using PyPDF. The text is filtered and split into smaller chunks.
2. **Embedding & Indexing**: Each chunk is embedded using the `all-MiniLM-L6-v2` sentence transformer model (384-dimensional vectors) and upserted into a Pinecone serverless index named `medical-chatbot` using cosine similarity.
3. **Serving** (`app.py`): The Flask app loads the existing Pinecone index at startup and sets up a LangChain RAG chain — a retriever that fetches the top 3 most relevant chunks, feeds them into a `ChatGroq` LLM, and returns a response.
4. **Chat Interface**: Users interact through a simple web chat UI (`chat.html`). The `/get` endpoint handles POST requests from the frontend, invokes the RAG chain, and returns the answer.

## Prerequisites

- Python 3.8+
- A [Pinecone](https://www.pinecone.io/) account with an API key
- A [Groq](https://console.groq.com/) account with an API key

## Installation

1. **Clone the repository**
```bash
   git clone https://github.com/ratul-podder99/RAG-Based-Medical-Chatbot.git
   cd RAG-Based-Medical-Chatbot
```

2. **Create and activate a virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**

   Create a `.env` file in the project root:
```env
   PINECONE_API_KEY=your_pinecone_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
```

5. **Add your medical PDF documents**

   Place any medical PDF files into the `data/` directory. These will form the chatbot's knowledge base.

## Usage

### Step 1 — Build the Vector Index (run once)

This script reads all PDFs from `data/`, generates embeddings, and populates the Pinecone index.

```bash
python store_index.py
```

> This step only needs to be run once, or again whenever you update the documents in `data/`.

### Step 2 — Launch the Web Application

```bash
python app.py
```

The app will start on `http://0.0.0.0:8080`. Open your browser and navigate to `http://localhost:8080` to access the chat interface.

## Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `index_name` | `app.py`, `store_index.py` | `medical-chatbot` | Pinecone index name |
| `search_kwargs["k"]` | `app.py` | `3` | Number of document chunks retrieved per query |
| `model_name` | `app.py` | `llama-3.1-8b-instant` | Groq LLM model |
| `temperature` | `app.py` | `0.1` | LLM response temperature (lower = more deterministic) |
| `max_tokens` | `app.py` | `1024` | Maximum tokens in LLM response |
| Embedding dimension | `store_index.py` | `384` | Vector size (matches `all-MiniLM-L6-v2`) |
| Pinecone cloud/region | `store_index.py` | `aws / us-east-1` | Pinecone serverless deployment region |

## Dependencies
langchain==0.3.26
flask==3.1.1
sentence-transformers==4.1.0
pypdf==5.6.1
python-dotenv==1.1.0
langchain-pinecone==0.2.8
langchain-openai==0.3.24
langchain-community==0.3.26
langchain-groq

## Notes

- This chatbot is intended for **informational purposes only** and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
- The quality of responses depends directly on the quality and coverage of the PDF documents placed in the `data/` directory.
- The Pinecone index is created automatically if it does not already exist when running `store_index.py`.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).

## Author

**Ratul Podder** — [ratulpodder99@gmail.com](mailto:ratulpodder99@gmail.com)
