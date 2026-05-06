"""
RAG module

This file handles both:
1. Gold chunks creation (data preparation)
2. Retrieval logic (used later for RAG queries)

What is a chunk?

A chunk is a small piece of a long document.
Instead of sending a full article to a model, we split it into smaller parts (chunks).
Each chunk contains a limited number of words (e.g., ~200 words).

Chunks are used in RAG to:
- reduce prompt size (lower cost and latency)
- retrieve only the most relevant parts of a document
- improve answer accuracy by focusing on useful context

In practice:
1. Split article → chunks
2. Store chunks
3. Retrieve relevant chunks for a query
4. Send only those chunks to the model


Why are Gold chunks defined here?

Chunks are directly used for Retrieval-Augmented Generation (RAG).
They are not just a storage layer — they are the core input for semantic search.

So it is consistent to group:
- chunk creation (build_gold_chunks)
- chunk usage (retrieve)

Pipeline logic:
- Bronze → ingest.py
- Silver → preprocess.py
- Gold chunks → rag.py (this file)
- RAG retrieval → rag.py (later)

This keeps the pipeline simple and aligned with the project architecture.
"""

from pathlib import Path
import pandas as pd
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "silver_articles.parquet"
GOLD_CHUNKS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_chunks.parquet"
GOLD_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_embeddings.npy"
GOLD_EMBEDDING_METADATA_PATH = PROJECT_ROOT / "data" / "gold" / "gold_embedding_metadata.parquet"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "gold" / "faiss_index.bin"

MAX_WORDS_PER_CHUNK = 220
MIN_CHARS_PER_CHUNK = 100


def approx_tokens(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return max(1, int(len(text.split()) * 1.3))


def chunk_text(text: str, max_words: int = MAX_WORDS_PER_CHUNK) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    words = text.split()
    chunks = []

    for start in range(0, len(words), max_words):
        chunk = " ".join(words[start:start + max_words]).strip()

        if len(chunk) >= MIN_CHARS_PER_CHUNK:
            chunks.append(chunk)

    return chunks


def build_gold_chunks(df_silver: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"id", "article_clean"}
    missing = required_columns - set(df_silver.columns)

    if missing:
        raise ValueError(f"Missing columns in Silver data: {missing}")

    records = []

    for _, row in df_silver.iterrows():
        article_id = str(row["id"])
        chunks = chunk_text(row["article_clean"])

        for chunk_index, chunk in enumerate(chunks):
            records.append({
                "chunk_id": f"{article_id}_chunk_{chunk_index}",
                "id": article_id,
                "chunk_text": chunk,
                "chunk_index": chunk_index,
                "approx_tokens": approx_tokens(chunk),
            })

    gold_chunks = pd.DataFrame(records)

    if gold_chunks.empty:
        raise ValueError("Gold chunks table is empty.")

    if gold_chunks["chunk_id"].duplicated().any():
        raise ValueError("Duplicate chunk_id found.")

    return gold_chunks


def build_embeddings(chunks_df: pd.DataFrame) -> None:
    """
    Build embeddings from Gold chunks.

    Each chunk_text is converted into a numerical vector.
    These vectors will later be used to find the chunks most similar
    to a user question.
    """

    if "chunk_text" not in chunks_df.columns:
        raise ValueError("Missing column: chunk_text")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = model.encode(
        chunks_df["chunk_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings, dtype="float32")

    GOLD_EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    np.save(GOLD_EMBEDDINGS_PATH, embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    metadata = chunks_df[["chunk_id", "id", "chunk_index", "approx_tokens"]].copy()
    metadata.to_parquet(
        GOLD_EMBEDDING_METADATA_PATH,
        index=False,
        compression="snappy"
    )

    print(f"Embeddings saved to: {GOLD_EMBEDDINGS_PATH}")
    print(f"Embedding metadata saved to: {GOLD_EMBEDDING_METADATA_PATH}")
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"FAISS vectors: {index.ntotal}")

def retrieve(query: str, k: int = 5) -> pd.DataFrame:
    """
    Retrieve the most relevant chunks for a user query using FAISS similarity search.
    """

    if not GOLD_CHUNKS_PATH.exists():
        raise FileNotFoundError("Gold chunks file not found.")

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index file not found.")

    chunks_df = pd.read_parquet(GOLD_CHUNKS_PATH)

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, k)

    retrieved = chunks_df.iloc[indices[0]].copy()

    retrieved["score"] = scores[0]

    return retrieved

def build_rag_prompt(question: str, retrieved_chunks: pd.DataFrame) -> str:
    """
    Build a compact RAG prompt using retrieved chunks.
    """

    context = "\n\n".join(retrieved_chunks["chunk_text"].tolist())

    prompt = f"""
You are a biomedical research assistant.
Answer only from the provided context.
If the answer is not present in the context, say so.

Context:
{context}

Question:
{question}

Answer:
""".strip()

    return prompt

def main():
    if not SILVER_PATH.exists():
        raise FileNotFoundError(
            f"Silver file not found: {SILVER_PATH}. Run preprocess.py first."
        )

    df_silver = pd.read_parquet(SILVER_PATH)

    print("\nSilver DataFrame tail:")
    print(df_silver.tail())

    # Check that the new article exists in Silver
    new_article_silver = df_silver[df_silver["id"].astype(str) == "new_article_001"]

    print("\nNew article in Silver:")
    print(new_article_silver)

    print("\nBuilding Gold chunks...")

    gold_chunks = build_gold_chunks(df_silver)

    GOLD_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    gold_chunks.to_parquet(
        GOLD_CHUNKS_PATH,
        index=False,
        compression="snappy"
    )

    print(f"Gold chunks saved to: {GOLD_CHUNKS_PATH}")
    print(f"Rows: {len(gold_chunks)}")

    print("\nGold chunks head:")
    print(gold_chunks.head())

    print("\nGold chunks tail:")
    print(gold_chunks.tail())

    # Check chunks created from the new article
    new_article_chunks = gold_chunks[
        gold_chunks["id"].astype(str) == "new_article_001"
        ]

    print("\nChunks generated for new_article_001:")
    print(new_article_chunks)

    build_embeddings(gold_chunks)


    print("\nTesting retrieval...")

    query = "What are the effects of hypertension on cardiovascular disease?"

    retrieved_chunks = retrieve(query, k=3)

    print("\nRetrieved chunks:")
    print(
        retrieved_chunks[
            [
                "chunk_id",
                "id",
                "chunk_index",
                "score",
            ]
        ]
    )

    print("\nTop retrieved chunk text:")
    print(retrieved_chunks.iloc[0]["chunk_text"])

    print("\nBuilding RAG prompt...")

    question = "What are the main cardiovascular outcomes mentioned in the retrieved context?"

    rag_prompt = build_rag_prompt(question, retrieved_chunks)

    print("\nRAG prompt preview:")
    print(rag_prompt[:1500])

if __name__ == "__main__":
    main()