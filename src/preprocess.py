"""
Understanding:
 - LLMs (like GPT) don’t read “characters” or “words” — they read tokens
 - Tokens ≠ words (they’re smaller chunks, often ~0.75 words on average in English)
 So we need to convert words to token: * 1.3 → converts words → tokens (approximation)
"""

from pathlib import Path
import pandas as pd

# SAME ROOT LOGIC AS ingest.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "bronze_articles.parquet"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "silver_articles.parquet"
SILVER_PATH_CSV = PROJECT_ROOT / "data" / "silver" / "silver_articles.csv"

def approx_tokens(text: str) -> int:
    """
    Rough token estimation.
    Assumes ~1.3 tokens per word (English average).
    Used for quick cost and size estimation.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    return max(1, int(len(text.split()) * 1.3))


def clean_text(text: str) -> str:
    """
    Normalize whitespace in text.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def validate_bronze(df: pd.DataFrame) -> None:
    """
    Ensure required columns exist and are valid.
    """
    required = {"id", "article", "abstract"}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["id"].isna().any():
        raise ValueError("Some rows have missing IDs.")

    if df["article"].isna().any():
        raise ValueError("Some rows have missing article text.")


def build_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform Bronze data into Silver layer.
    """
    validate_bronze(df)

    silver = pd.DataFrame()

    silver["id"] = df["id"].astype(str)
    silver["article_clean"] = df["article"].apply(clean_text)
    silver["abstract_clean"] = df["abstract"].apply(clean_text)

    silver["article_chars"] = silver["article_clean"].str.len()
    silver["abstract_chars"] = silver["abstract_clean"].str.len()
    silver["approx_tokens"] = silver["article_clean"].apply(approx_tokens)

    # Basic validation rules
    silver = silver[silver["id"].str.strip() != ""]
    silver = silver[silver["article_clean"].str.strip() != ""]
    silver = silver[silver["approx_tokens"] > 0]

    # Remove duplicates
    silver = silver.drop_duplicates(subset=["id"], keep="first")

    return silver


def main():
    SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Bronze data...")
    df_bronze = pd.read_parquet(BRONZE_PATH)

    print("Building Silver layer...")
    df_silver = build_silver(df_bronze)

    df_silver.to_parquet(
        SILVER_PATH,
        index=False,
        compression="snappy"
    )

    df_silver.to_csv(
        SILVER_PATH_CSV,
        index=False,
        sep = ";"
    )

    print(f"Saved Silver dataset to {SILVER_PATH}")
    print(f"Rows: {len(df_silver)}")


if __name__ == "__main__":
    main()