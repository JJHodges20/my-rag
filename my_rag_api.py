from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from my_rag import (
    OLLAMA_MODEL,
    answer_question,
    collection,
    ingest_documents,
)


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="My RAG API",
    description=(
        "A FastAPI interface for a guarded "
        "Retrieval-Augmented Generation pipeline."
    ),
    version="1.0.0",
)


# ============================================
# CORS
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# SETTINGS
# ============================================

OLLAMA_TAGS_URL = (
    "http://localhost:11434/api/tags"
)


# ============================================
# PYDANTIC SCHEMAS
# ============================================

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the RAG system.",
        examples=[
            "How does Streamlit preserve values between reruns?"
        ],
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(
        cls,
        value,
    ):
        if not value.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        return value.strip()


class AskResponse(BaseModel):
    answer: str

    sources: list[str]

    confidence: Literal[
        "high",
        "medium",
        "low",
    ]

    chunks_retrieved: int


class IngestResponse(BaseModel):
    message: str

    document_count: int

    chunk_count: int


class StatsResponse(BaseModel):
    document_count: int

    chunk_count: int

    ollama_model: str

    embedding_model: str

    collection_name: str


class HealthResponse(BaseModel):
    status: str

    chromadb: str

    ollama: str

    ollama_model: str


# ============================================
# HELPER FUNCTIONS
# ============================================

def ollama_is_running():
    """
    Check whether the local Ollama service
    is responding.
    """

    try:

        response = requests.get(
            OLLAMA_TAGS_URL,
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


def get_document_count():
    """
    Count unique source documents currently
    stored in ChromaDB.
    """

    if collection.count() == 0:
        return 0


    results = collection.get(
        include=[
            "metadatas"
        ]
    )


    metadatas = results.get(
        "metadatas",
        [],
    )


    sources = {
        metadata["source"]
        for metadata in metadatas
        if metadata
        and "source" in metadata
    }


    return len(sources)


# ============================================
# POST /ASK
# ============================================

@app.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the RAG system a question",
)
def ask_question(
    request: AskRequest,
):
    """
    Retrieve relevant document chunks and
    generate a grounded answer with Ollama.
    """

    # ----------------------------------------
    # CHECK FOR DOCUMENTS
    # ----------------------------------------

    if collection.count() == 0:

        return AskResponse(
            answer=(
                "No documents have been ingested yet. "
                "Use POST /ingest before asking questions."
            ),
            sources=[],
            confidence="low",
            chunks_retrieved=0,
        )


    # ----------------------------------------
    # CHECK OLLAMA
    # ----------------------------------------

    if not ollama_is_running():

        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running. "
                "Start Ollama and try again."
            ),
        )


    # ----------------------------------------
    # RUN RAG PIPELINE
    # ----------------------------------------

    result = answer_question(
        request.question
    )


    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        chunks_retrieved=(
            result["chunks_retrieved"]
        ),
    )


# ============================================
# POST /INGEST
# ============================================

@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into ChromaDB",
)
def ingest():
    """
    Load the text files from the docs folder,
    chunk them, and store them in ChromaDB.
    """

    try:

        ingest_documents()

        document_count = (
            get_document_count()
        )

        chunk_count = (
            collection.count()
        )


        return IngestResponse(
            message=(
                "Documents successfully ingested."
            ),
            document_count=document_count,
            chunk_count=chunk_count,
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Document ingestion failed: "
                f"{error}"
            ),
        )


# ============================================
# GET /STATS
# ============================================

@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="View RAG system statistics",
)
def get_stats():
    """
    Return information about the stored
    knowledge base and configured models.
    """

    try:

        document_count = (
            get_document_count()
        )

        chunk_count = (
            collection.count()
        )


        return StatsResponse(
            document_count=document_count,
            chunk_count=chunk_count,
            ollama_model=OLLAMA_MODEL,
            embedding_model=(
                "all-MiniLM-L6-v2"
            ),
            collection_name=(
                collection.name
            ),
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Could not read RAG stats: "
                f"{error}"
            ),
        )


# ============================================
# GET /HEALTH
# ============================================

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check RAG system health",
)
def health_check():
    """
    Check whether ChromaDB and Ollama
    are available.
    """

    # ----------------------------------------
    # CHECK CHROMADB
    # ----------------------------------------

    try:

        collection.count()

        chromadb_status = "healthy"

    except Exception:

        raise HTTPException(
            status_code=503,
            detail=(
                "ChromaDB is not accessible."
            ),
        )


    # ----------------------------------------
    # CHECK OLLAMA
    # ----------------------------------------

    if not ollama_is_running():

        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running."
            ),
        )


    return HealthResponse(
        status="healthy",
        chromadb=chromadb_status,
        ollama="healthy",
        ollama_model=OLLAMA_MODEL,
    )