from pathlib import Path

import chromadb
import requests
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


# ============================================
# SETTINGS
# ============================================

DOCS_FOLDER = Path("docs")

CHROMA_FOLDER = "chroma_data"

COLLECTION_NAME = "course_knowledge"

OLLAMA_URL = "http://localhost:11434/api/chat"

OLLAMA_MODEL = "llama3.2"


# ============================================
# EMBEDDING FUNCTION
# ============================================

embedding_function = (
    SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)


# ============================================
# CHROMADB
# ============================================

client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)


collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


# ============================================
# LOAD DOCUMENTS
# ============================================

def load_documents():
    """
    Load all .txt files from the docs folder.
    """

    documents = []

    for file_path in DOCS_FOLDER.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "source": file_path.name,
                "text": text,
            }
        )

    return documents


# ============================================
# PARAGRAPH CHUNKING
# ============================================

def chunk_document(text):
    """
    Split a document into paragraph-based chunks.
    """

    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    return chunks


# ============================================
# INGEST DOCUMENTS
# ============================================

def ingest_documents():
    """
    Load documents, chunk them by paragraph,
    and store them in persistent ChromaDB.
    """

    documents = load_documents()

    total_chunks = 0


    for document in documents:

        source = document["source"]

        chunks = chunk_document(
            document["text"]
        )


        # Remove old chunks for this source
        # before inserting the current version.

        try:
            collection.delete(
                where={
                    "source": source
                }
            )
        except Exception:
            pass


        for index, chunk in enumerate(chunks):

            chunk_id = (
                f"{source}-chunk-{index}"
            )


            collection.upsert(
                ids=[
                    chunk_id
                ],
                documents=[
                    chunk
                ],
                metadatas=[
                    {
                        "source": source,
                        "chunk_index": index,
                    }
                ],
            )


            total_chunks += 1


    print(
        f"Ingested {len(documents)} documents "
        f"and {total_chunks} chunks."
    )


# ============================================
# RETRIEVAL
# ============================================

def retrieve(question, top_k=3):
    """
    Search ChromaDB and return the top matching chunks.
    """

    results = collection.query(
        query_texts=[
            question
        ],
        n_results=top_k,
    )


    retrieved_chunks = []


    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]


    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):

        retrieved_chunks.append(
            {
                "text": document,
                "source": metadata["source"],
                "chunk_index": metadata[
                    "chunk_index"
                ],
                "distance": distance,
            }
        )


    return retrieved_chunks


# ============================================
# BUILD RAG PROMPT
# ============================================

def build_prompt(
    question,
    retrieved_chunks,
):
    """
    Build a grounded RAG prompt using retrieved chunks.
    """

    context_sections = []


    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        source_label = (
            f"Source {index}: "
            f"{chunk['source']} "
            f"(chunk {chunk['chunk_index']})"
        )


        context_sections.append(
            f"[{source_label}]\n"
            f"{chunk['text']}"
        )


    context = "\n\n".join(
        context_sections
    )


    system_prompt = """
You are a course study assistant using Retrieval-Augmented Generation.

Answer the user's question using ONLY the retrieved context provided.

Rules:
- Do not use outside knowledge.
- If the retrieved context does not contain enough information to answer the question, clearly say so.
- Do not invent facts.
- Cite the source or sources used in your answer.
- Use citations such as [Source 1] or [Source 2].
- Keep the answer clear and concise.
""".strip()


    user_prompt = f"""
RETRIEVED CONTEXT
-----------------
{context}


USER QUESTION
-------------
{question}


Answer the question using only the retrieved context and include source citations.
""".strip()


    return (
        system_prompt,
        user_prompt,
    )


# ============================================
# OLLAMA GENERATION
# ============================================

def generate_answer(
    question,
    retrieved_chunks,
):
    """
    Send the RAG prompt to Ollama.
    """

    system_prompt, user_prompt = (
        build_prompt(
            question,
            retrieved_chunks,
        )
    )


    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
    }


    try:

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )


        response.raise_for_status()


        data = response.json()


        return data[
            "message"
        ][
            "content"
        ]


    except requests.exceptions.ConnectionError:

        return (
            "Error: Could not connect to Ollama. "
            "Make sure Ollama is installed and running."
        )


    except requests.exceptions.Timeout:

        return (
            "Error: Ollama took too long to respond."
        )


    except requests.RequestException as error:

        return (
            f"Error communicating with Ollama: "
            f"{error}"
        )


# ============================================
# DISPLAY RETRIEVED CHUNKS
# ============================================

def display_retrieved_chunks(
    retrieved_chunks,
):

    print(
        "\nRetrieved chunks:"
    )


    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        print(
            f"\n{index}. "
            f"{chunk['source']} "
            f"(chunk {chunk['chunk_index']})"
        )

        print(
            f"   Distance: "
            f"{chunk['distance']:.4f}"
        )

        print(
            f"   {chunk['text']}"
        )


# ============================================
# MAIN
# ============================================

def main():

    print(
        "Building RAG knowledge base..."
    )


    ingest_documents()


    print(
        "\nRAG system ready."
    )

    print(
        "Ask a question or type 'quit' to exit."
    )


    while True:

        question = input(
            "\nQuestion: "
        ).strip()


        if question.lower() == "quit":

            print(
                "Goodbye!"
            )

            break


        if not question:

            print(
                "Please enter a question."
            )

            continue


        # ------------------------------------
        # RETRIEVE
        # ------------------------------------

        retrieved_chunks = retrieve(
            question,
            top_k=3,
        )


        # ------------------------------------
        # DISPLAY RETRIEVAL
        # ------------------------------------

        display_retrieved_chunks(
            retrieved_chunks
        )


        # ------------------------------------
        # GENERATE
        # ------------------------------------

        print(
            "\nGenerating answer..."
        )


        answer = generate_answer(
            question,
            retrieved_chunks,
        )


        # ------------------------------------
        # DISPLAY ANSWER
        # ------------------------------------

        print(
            "\nAnswer:"
        )

        print(
            answer
        )


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()