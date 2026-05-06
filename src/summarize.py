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

def build_p1_long_prompt(article_text: str) -> str:
    """
    P1: Long prompt using the full article.
    Higher input token cost.
    """
    return f"""
Summarize the following biomedical article in an abstract-style format.

Article:
{article_text}

Summary:
""".strip()


def build_p2_compact_prompt(article_text: str, max_words: int = 500) -> str:
    """
    P2: Compact prompt using only the beginning of the article.
    Lower input token cost.
    """
    short_article = " ".join(article_text.split()[:max_words])

    return f"""
Write a concise biomedical abstract-style summary.

Text:
{short_article}

Summary:
""".strip()

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

        has_reference = isinstance(reference_abstract, str) and reference_abstract.strip() != ""

        prompt_versions = [
            ("P1_long_prompt", build_p1_long_prompt(article_text)),
            ("P2_compact_prompt", build_p2_compact_prompt(article_text)),
        ]

        for prompt_version, prompt in prompt_versions:
            start = time.time()
            summary = simple_summary(article_text)
            latency = time.time() - start

            quality_score = simple_quality_score(summary, reference_abstract)

            if has_reference:
                notes = "Reference abstract available; baseline summary compared with reference. No API used."
            else:
                notes = "No reference abstract available; baseline summary generated only. No API used."

            records.append({
                "id": article_id,
                "model": "baseline_truncate",
                "prompt_version": prompt_version,
                "generated_summary": summary,
                "reference_abstract": reference_abstract,
                "has_reference_abstract": has_reference,
                "latency_s": latency,
                "input_tokens": approx_tokens(prompt),
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
                    "prompt_version",
                    "has_reference_abstract",
                    "input_tokens",
                    "output_tokens",
                    "quality_score",
                    "notes",
                ]
            ]
        )

    print("\nPrompt version token comparison:")
    print(
        df_summary.groupby("prompt_version")[
            ["input_tokens", "output_tokens", "latency_s", "quality_score"]
        ].mean()
    )

if __name__ == "__main__":
    main()