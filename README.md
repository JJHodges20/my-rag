# My RAG Pipeline

A Python project that builds a complete Retrieval-Augmented Generation (RAG) pipeline using local documents, ChromaDB, SentenceTransformers, and Ollama.

The application loads text files, splits them into chunks, stores them in a persistent vector database, retrieves the most relevant chunks for a user question, and sends the retrieved context to a local Ollama model for a grounded response.

## Features

- Loads documents from a `docs/` folder
- Splits documents into paragraph-based chunks
- Creates embeddings with `all-MiniLM-L6-v2`
- Stores chunks in persistent ChromaDB
- Retrieves the top 3 relevant chunks for each question
- Displays retrieved chunks before generation
- Builds a grounded RAG prompt with source citations
- Generates responses with a local Ollama model
- Handles Ollama connection errors gracefully
- Runs interactively until the user types `quit`

## Technologies

- Python
- ChromaDB
- sentence-transformers
- Ollama
- requests
- Local LLMs

## Project Structure

```text
my-rag/
├── docs/
│   ├── fastapi.txt
│   ├── databases.txt
│   ├── web_development.txt
│   ├── streamlit.txt
│   ├── semantic_search.txt
│   └── rag.txt
├── my_rag.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Make sure Ollama is installed and that the model used in `my_rag.py` is available:

```powershell
ollama list
```

If needed:

```powershell
ollama pull llama3.2
```

## Running the Project

Run:

```powershell
python my_rag.py
```

Then enter questions at the prompt.

Example:

```text
Question: How does Streamlit remember values after reruns?
```

Type:

```text
quit
```

to exit.

## Purpose

This project was created to combine document ingestion, chunking, embeddings, vector search, prompt assembly, and local LLM generation into one working RAG pipeline.