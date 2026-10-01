# Guarded RAG Pipeline

A Python project that extends a basic Retrieval-Augmented Generation (RAG) pipeline with reliability features designed to reduce weak retrievals and unsupported answers.

The application loads local documents, stores paragraph chunks in ChromaDB, retrieves relevant context for a user question, applies distance-based guardrails, and uses a local Ollama model to generate grounded responses with citations.

## Features

- Loads text documents from a `docs/` folder
- Splits documents into paragraph-based chunks
- Stores embeddings in persistent ChromaDB
- Retrieves candidate chunks for each question
- Filters weak results using a configurable distance threshold
- Assigns response confidence levels
- Uses a strengthened system prompt to reduce hallucinations
- Requires source citations
- Returns results in a structured dictionary
- Handles cases where no relevant context is found
- Includes four built-in test query types
- Supports interactive questioning after the tests

## Guardrails

### 1. Distance Threshold

Only chunks with a ChromaDB distance below the configured threshold are allowed into the final prompt.

Default:

```python
DISTANCE_THRESHOLD = 1.0
```

Lower distances represent stronger matches.

### 2. Confidence Levels

Confidence is based on the best retrieved chunk:

```text
High   = distance below 0.5
Medium = distance below 1.0
Low    = distance 1.0 or higher, or no relevant chunks
```

### 3. Grounded System Prompt

The Ollama model is instructed to:

- Use only the retrieved context
- Never invent missing information
- Say when the answer is unknown
- Always cite supporting sources
- Avoid invented citations

### 4. Structured Responses

Each result is returned as a dictionary:

```python
{
    "answer": "...",
    "sources": ["streamlit.txt"],
    "confidence": "high",
    "chunks_retrieved": 2
}
```

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
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

- Python
- ChromaDB
- SentenceTransformers
- Ollama
- requests
- Local LLMs

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

Make sure Ollama is installed and running:

```powershell
ollama list
```

If needed, pull the model used by the project:

```powershell
ollama pull llama3.2
```

## Running the Project

Run:

```powershell
python my_rag.py
```

The program first tests four types of questions:

- In-scope
- Partially in-scope
- Out-of-scope
- Ambiguous

For each query, the program displays the retrieved chunks, their distances, whether they passed the threshold, and the final structured response.

After the tests finish, the program enters interactive mode.

Type:

```text
quit
```

to exit.

## Purpose

This project builds on a basic RAG pipeline by adding reliability features such as retrieval thresholds, confidence levels, stronger grounding instructions, and structured output. These guardrails help prevent weak context from being passed to the language model and make the system easier to evaluate and integrate into larger applications.