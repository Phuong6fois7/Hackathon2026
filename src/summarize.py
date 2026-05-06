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

def simple_quality_score(generated_summary: str, reference_abstract: str):
    """
    Basic comparison between generated summary and reference abstract.
    This is a simple word-overlap score, not a final NLP metric.
    """
    if not isinstance(reference_abstract, str) or not reference_abstract.strip():
        return None

    generated_words = set(generated_summary.lower().split())
    reference_words = set(reference_abstract.lower().split())

    if len(reference_words) == 0:
        return None

    overlap = generated_words.intersection(reference_words)
    return round(len(overlap) / len(reference_words), 4)

def build_summaries(df_silver: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in df_silver.iterrows():
        article_id = str(row["id"])
        article_text = row["article_clean"]
        reference_abstract = row["abstract_clean"]

        start = time.time()
        summary = simple_summary(article_text)
        latency = time.time() - start

        has_reference = isinstance(reference_abstract, str) and reference_abstract.strip() != ""

        quality_score = simple_quality_score(summary, reference_abstract)

        if has_reference:
            notes = "Reference abstract available; generated summary compared with reference abstract."
        else:
            notes = "No reference abstract available; generated summary only."

        records.append({
            "id": article_id,
            "model": "baseline_truncate",
            "prompt_version": "P0_simple",
            "generated_summary": summary,
            "reference_abstract": reference_abstract,
            "has_reference_abstract": has_reference,
            "latency_s": latency,
            "input_tokens": approx_tokens(article_text),
            "output_tokens": approx_tokens(summary),
            "quality_score": quality_score,
            "notes": notes,
        })

    return pd.DataFrame(records)

def main():
    if not SILVER_PATH.exists():
        raise FileNotFoundError("Run preprocess.py first")

    df = pd.read_parquet(SILVER_PATH)

    print("Building summaries...")
    df_summary = build_summaries(df)  # small sample

    GOLD_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_summary.to_parquet(GOLD_SUMMARY_PATH, index=False, compression="snappy")

    print(f"Saved summaries to {GOLD_SUMMARY_PATH}")
    print(f"Rows: {len(df_summary)}")

    print("\nGold summaries head:")
    print(df_summary.head())

    print("\nGold summaries tail:")
    print(df_summary.tail())

    print("\nComparison for new_article_001:")
    new_article_summary = df_summary[
        df_summary["id"].astype(str) == "new_article_001"
        ]

    if new_article_summary.empty:
        print("new_article_001 not found in Gold summaries.")
    else:
        print(
            new_article_summary[
                [
                    "id",
                    "has_reference_abstract",
                    "generated_summary",
                    "reference_abstract",
                    "quality_score",
                    "notes",
                ]
            ]
        )


if __name__ == "__main__":
    main()