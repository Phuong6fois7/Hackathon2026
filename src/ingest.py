from datasets import load_dataset
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import json


DATASET_NAME = "ccdv/pubmed-summarization"
DATASET_CONFIG = "document"
SOURCE_SPLIT = "train"
DATASET_SLICE = "train[:100]"   # Mode DEV
# DATASET_SLICE = "train"       # mode complet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
OUTPUT_FILE = BRONZE_DIR / "bronze_articles.parquet"
NEW_ARTICLE_PATH = BRONZE_DIR / "new_article.json"

def create_bronze_articles() -> pd.DataFrame:
    """
    Load raw PubMed articles and create the Bronze articles table.

    Expected columns:
    id, article, abstract, source_split, ingestion_ts
    """

    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split=DATASET_SLICE
    )

    df = ds.to_pandas()

    required_columns = ["article", "abstract"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Keep only rows with non-empty article
    df = df.dropna(subset=["article"])
    df["article"] = df["article"].astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)

    df = df[df["article"].str.strip() != ""].copy()

    # Simple readable ID
    df = df.reset_index(drop=True)
    df["id"] = df.index.map(lambda x: f"article_{x}")

    # # # Other way: id is also index
    # # # Reset index propre
    # df = df.reset_index(drop=True)
    # # ID numérique simple
    # df["id"] = df.index


    # Metadata
    df["source_split"] = SOURCE_SPLIT
    df["ingestion_ts"] = datetime.now(timezone.utc).isoformat()

    bronze_articles = df[
        [
            "id",
            "article",
            "abstract",
            "source_split",
            "ingestion_ts",
        ]
    ]

    validate_bronze_articles(bronze_articles)

    return bronze_articles


def validate_bronze_articles(df: pd.DataFrame) -> None:
    """
    Validation rule:
    No empty id; article is not empty.
    """

    expected_columns = [
        "id",
        "article",
        "abstract",
        "source_split",
        "ingestion_ts",
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df["id"].isna().any():
        raise ValueError("Validation failed: id contains null values.")

    if (df["id"].astype(str).str.strip() == "").any():
        raise ValueError("Validation failed: id contains empty values.")

    if df["article"].isna().any():
        raise ValueError("Validation failed: article contains null values.")

    if (df["article"].astype(str).str.strip() == "").any():
        raise ValueError("Validation failed: article contains empty values.")


def save_bronze_articles(df: pd.DataFrame) -> None:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Bronze articles saved to: {OUTPUT_FILE}")
    print(f"Rows saved: {len(df)}")


def append_new_article_to_bronze() -> None:
    """
    Append a simulated new article to the Bronze layer.

    Steps:
    - read a raw JSON article from data/bronze/new_article.json
    - validate required fields
    - load existing Bronze Parquet data
    - avoid duplicate IDs
    - append the new article
    - save Bronze again
    """

    if not NEW_ARTICLE_PATH.exists():
        raise FileNotFoundError(f"New article file not found: {NEW_ARTICLE_PATH}")

    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Bronze file not found: {OUTPUT_FILE}. Run ingest.py first."
        )

    with open(NEW_ARTICLE_PATH, "r", encoding="utf-8") as file:
        new_article = json.load(file)

    required_fields = ["id", "article"]

    for field in required_fields:
        if field not in new_article:
            raise ValueError(f"Missing required field in new article: {field}")

    new_id = str(new_article["id"]).strip()
    article_text = str(new_article["article"]).strip()
    abstract_text = str(new_article.get("abstract", "")).strip()

    if not new_id:
        raise ValueError("New article id is empty.")

    if not article_text:
        raise ValueError("New article text is empty.")

    df_bronze = pd.read_parquet(OUTPUT_FILE)

    if new_id in df_bronze["id"].astype(str).values:
        print(f"Article already exists in Bronze: {new_id}")
        return

    new_row = pd.DataFrame([
        {
            "id": new_id,
            "article": article_text,
            "abstract": abstract_text,
            "source_split": "new_article",
            "ingestion_ts": datetime.now(timezone.utc).isoformat(),
        }
    ])

    df_updated = pd.concat([df_bronze, new_row], ignore_index=True)

    validate_bronze_articles(df_updated)

    df_updated.to_parquet(OUTPUT_FILE, index=False)

    print("\nUpdated Bronze DataFrame tail:")
    print(df_updated.tail())

    print(f"New article appended to Bronze: {new_id}")
    print(f"Total Bronze rows: {len(df_updated)}")


def main():
    # bronze_articles = create_bronze_articles()
    # save_bronze_articles(bronze_articles)
    # print(bronze_articles.head())

    # Test append new article (desactivate all bronze_articles first, then append_new_article_to_bronze())
     append_new_article_to_bronze()

if __name__ == "__main__":
    main()
