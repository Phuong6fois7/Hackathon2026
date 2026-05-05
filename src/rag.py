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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "silver_articles.parquet"
GOLD_CHUNKS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_chunks.parquet"

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


def main():
    if not SILVER_PATH.exists():
        raise FileNotFoundError(
            f"Silver file not found: {SILVER_PATH}. Run preprocess.py first."
        )

    df_silver = pd.read_parquet(SILVER_PATH)

    gold_chunks = build_gold_chunks(df_silver)

    GOLD_CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    gold_chunks.to_parquet(
        GOLD_CHUNKS_PATH,
        index=False,
        compression="snappy"
    )

    print(f"Gold chunks saved to: {GOLD_CHUNKS_PATH}")
    print(f"Rows: {len(gold_chunks)}")
    print(gold_chunks.head())


if __name__ == "__main__":
    main()