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

DISTANCE_THRESHOLD = 1.0


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

    return [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]


# ============================================
# INGEST DOCUMENTS
# ============================================

def ingest_documents():
    """
    Load documents, split them into paragraphs,
    and store them in persistent ChromaDB.
    """

    documents = load_documents()

    total_chunks = 0


    for document in documents:

        source = document["source"]

        chunks = chunk_document(
            document["text"]
        )


        # Remove older chunks from this file
        # before adding the current version.
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

def retrieve(
    question,
    top_k=5,
):
    """
    Retrieve candidate chunks from ChromaDB.

    We request more than 3 initially so the
    threshold can filter weak matches.
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
# GUARDRAIL 1
# DISTANCE THRESHOLD
# ============================================

def filter_chunks(
    chunks,
    threshold=DISTANCE_THRESHOLD,
):
    """
    Keep only chunks with distance below
    the configured threshold.
    """

    filtered = [
        chunk
        for chunk in chunks
        if chunk["distance"] < threshold
    ]

    return filtered


# ============================================
# GUARDRAIL 2
# CONFIDENCE LEVEL
# ============================================

def get_confidence(chunks):
    """
    Determine confidence using the best distance.

    Smaller Chroma distance = stronger match.
    """

    if not chunks:
        return "low"


    best_distance = min(
        chunk["distance"]
        for chunk in chunks
    )


    if best_distance < 0.5:
        return "high"

    elif best_distance < 1.0:
        return "medium"

    else:
        return "low"


# ============================================
# BUILD RAG PROMPT
# ============================================

def build_prompt(
    question,
    retrieved_chunks,
):
    """
    Build a guarded RAG prompt.
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
You are a grounded course study assistant.

You must follow these guardrails:

- Answer using ONLY the retrieved context provided.
- Never make up facts or add information from outside the context.
- If the context does not contain enough information, say:
  "I don't know based on the provided context."
- If you are unsure, say that you are unsure.
- Always cite the source or sources used.
- Use citations such as [Source 1] or [Source 2].
- Do not invent citations.
- Keep the response clear and concise.
""".strip()


    user_prompt = f"""
RETRIEVED CONTEXT
-----------------
{context}


USER QUESTION
-------------
{question}


Answer using only the retrieved context.

Remember:
- Do not make up missing information.
- Say "I don't know based on the provided context."
  if the answer is not supported.
- Cite your sources.
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
    Send the grounded RAG prompt to Ollama.
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
# GUARDRAIL 4
# STRUCTURED RESPONSE
# ============================================

def answer_question(question):
    """
    Run retrieval, filtering, confidence scoring,
    generation, and structured response creation.
    """

    candidate_chunks = retrieve(
        question,
        top_k=5,
    )


    filtered_chunks = filter_chunks(
        candidate_chunks,
        threshold=DISTANCE_THRESHOLD,
    )


    confidence = get_confidence(
        filtered_chunks
    )


    # ----------------------------------------
    # NO RELEVANT INFORMATION
    # ----------------------------------------

    if not filtered_chunks:

        return {
            "answer": (
                "No relevant information was found "
                "in the knowledge base."
            ),
            "sources": [],
            "confidence": "low",
            "chunks_retrieved": 0,
        }


    # Keep only the best 3 chunks after filtering
    filtered_chunks = filtered_chunks[:3]


    answer = generate_answer(
        question,
        filtered_chunks,
    )


    sources = list(
        dict.fromkeys(
            chunk["source"]
            for chunk in filtered_chunks
        )
    )


    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "chunks_retrieved": len(
            filtered_chunks
        ),
    }


# ============================================
# DISPLAY RETRIEVED CHUNKS
# ============================================

def display_chunks(question):
    """
    Display candidate retrieval results so
    the guardrail behavior can be inspected.
    """

    chunks = retrieve(
        question,
        top_k=5,
    )


    print(
        "\nCandidate chunks:"
    )


    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        passes = (
            chunk["distance"]
            < DISTANCE_THRESHOLD
        )


        status = (
            "PASS"
            if passes
            else "FILTERED"
        )


        print(
            f"\n{index}. "
            f"[{status}] "
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
# TEST QUERIES
# ============================================

test_queries = {

    "IN-SCOPE":
        "How does Streamlit preserve values between reruns?",

    "PARTIALLY IN-SCOPE":
        "What database should I use for a large production FastAPI app?",

    "OUT-OF-SCOPE":
        "Who won the Super Bowl in 2025?",

    "AMBIGUOUS":
        "How do I make my application better?",
}


# ============================================
# RUN TESTS
# ============================================

def run_tests():

    print(
        "\nRunning RAG guardrail tests..."
    )


    for query_type, question in (
        test_queries.items()
    ):

        print(
            "\n" + "=" * 80
        )

        print(
            f"{query_type}"
        )

        print(
            f'Question: "{question}"'
        )


        display_chunks(
            question
        )


        response = answer_question(
            question
        )


        print(
            "\nStructured response:"
        )

        print(
            response
        )


# ============================================
# MAIN
# ============================================

def main():

    print(
        "Building guarded RAG knowledge base..."
    )


    ingest_documents()


    print(
        f"\nDistance threshold: "
        f"{DISTANCE_THRESHOLD}"
    )


    run_tests()


    print(
        "\n" + "=" * 80
    )

    print(
        "\nInteractive mode ready."
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


        display_chunks(
            question
        )


        response = answer_question(
            question
        )


        print(
            "\nStructured response:"
        )

        print(
            response
        )


# ============================================
# ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()