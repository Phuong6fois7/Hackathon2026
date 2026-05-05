"""Without API, test simple with AI output"""

from pathlib import Path
import pandas as pd
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "silver_articles.parquet"
GOLD_SUMMARY_PATH = PROJECT_ROOT / "data" / "gold" / "gold_summaries.parquet"


def approx_tokens(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return max(1, int(len(text.split()) * 1.3))


def simple_summary(text: str, max_words: int = 120) -> str:
    """
    Simple baseline: take first N words.
    (Used as a placeholder before real AI model)
    """
    if not isinstance(text, str):
        return ""

    words = text.split()
    return " ".join(words[:max_words])


def build_summaries(df_silver: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in df_silver.iterrows():
        article_id = str(row["id"])

        start = time.time()

        summary = simple_summary(row["article_clean"])

        latency = time.time() - start

        records.append({
            "id": article_id,
            "model": "baseline_truncate",
            "prompt_version": "P0_simple",
            "generated_summary": summary,
            "latency_s": latency,
            "input_tokens": approx_tokens(row["article_clean"]),
            "output_tokens": approx_tokens(summary),
        })

    return pd.DataFrame(records)


def main():
    if not SILVER_PATH.exists():
        raise FileNotFoundError("Run preprocess.py first")

    df = pd.read_parquet(SILVER_PATH)

    print("Building summaries...")
    df_summary = build_summaries(df.head(50))  # small sample

    GOLD_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_summary.to_parquet(GOLD_SUMMARY_PATH, index=False)

    print(f"Saved summaries to {GOLD_SUMMARY_PATH}")
    print(df_summary.head())


if __name__ == "__main__":
    main()