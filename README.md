# My RAG API

A Python project that builds a production-style Retrieval-Augmented Generation (RAG) system using ChromaDB, SentenceTransformers, Ollama, and FastAPI.

The project started as a local RAG pipeline and now includes a REST API that can ingest documents, answer grounded questions, report system statistics, and check the health of ChromaDB and Ollama.

## Features

- Loads text documents from a `docs/` folder
- Splits documents into paragraph-based chunks
- Creates embeddings with `all-MiniLM-L6-v2`
- Stores chunks in persistent ChromaDB
- Retrieves relevant chunks for user questions
- Filters weak matches with a distance threshold
- Assigns confidence levels
- Uses grounded system prompts to reduce hallucinations
- Includes source citations
- Returns structured RAG responses
- Provides a FastAPI interface
- Includes Swagger UI documentation
- Checks ChromaDB and Ollama health
- Handles common API errors gracefully
- Includes CORS middleware for frontend connectivity

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
├── chroma_data/
├── my_rag.py
├── my_rag_api.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

- Python
- FastAPI
- Pydantic
- Uvicorn
- ChromaDB
- SentenceTransformers
- Ollama
- requests
- Local LLMs

## RAG Guardrails

The RAG pipeline includes several reliability features.

### Distance Threshold

Only chunks below the configured ChromaDB distance threshold are used.

```python
DISTANCE_THRESHOLD = 1.0
```

Lower distances represent stronger semantic matches.

### Confidence Levels

Responses are assigned a confidence level based on the strongest retrieved chunk:

```text
High   = best distance below 0.5
Medium = best distance below 1.0
Low    = no relevant information found
```

### Grounded Prompt

The Ollama model is instructed to:

- Answer only from retrieved context
- Never invent missing information
- Say when it does not know
- Always cite supporting sources
- Avoid invented citations

## API Endpoints

### `POST /ask`

Accepts a question and returns a grounded RAG response.

Example request:

```json
{
  "question": "How does Streamlit preserve values between reruns?"
}
```

Example response:

```json
{
  "answer": "Streamlit uses session state to preserve values between reruns [Source 1].",
  "sources": [
    "streamlit.txt"
  ],
  "confidence": "high",
  "chunks_retrieved": 2
}
```

### `POST /ingest`

Loads the text files from the `docs/` folder, chunks them, and stores them in ChromaDB.

Example response:

```json
{
  "message": "Documents successfully ingested.",
  "document_count": 6,
  "chunk_count": 20
}
```

### `GET /stats`

Returns information about the current RAG system.

Example:

```json
{
  "document_count": 6,
  "chunk_count": 20,
  "ollama_model": "llama3.2",
  "embedding_model": "all-MiniLM-L6-v2",
  "collection_name": "course_knowledge"
}
```

### `GET /health`

Checks whether ChromaDB and Ollama are accessible.

Example:

```json
{
  "status": "healthy",
  "chromadb": "healthy",
  "ollama": "healthy",
  "ollama_model": "llama3.2"
}
```

If Ollama is unavailable, the API returns:

```text
503 Service Unavailable
```

## Validation and Error Handling

The API includes error handling for several common situations:

- Ollama not running → `503 Service Unavailable`
- ChromaDB unavailable → `503 Service Unavailable`
- No documents ingested → clear response from `/ask`
- Empty questions → `422 Unprocessable Entity`
- Document ingestion errors → `500 Internal Server Error`

Pydantic handles request validation automatically.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Make sure Ollama is installed and running:

```powershell
ollama list
```

If needed, pull the model:

```powershell
ollama pull llama3.2
```

## Running the API

Start the FastAPI server:

```powershell
python -m uvicorn my_rag_api:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

## Swagger UI

Open:

```text
http://127.0.0.1:8000/docs
```

Swagger UI can be used to test all four endpoints directly in the browser.

A useful testing order is:

```text
GET  /health
POST /ingest
GET  /stats
POST /ask
```

## Purpose

This project demonstrates how a local RAG pipeline can be turned into a reusable API service.

It combines document ingestion, embeddings, vector search, retrieval guardrails, confidence scoring, Ollama generation, validation, health checks, and FastAPI into one complete application that can later connect to a frontend or other software system.