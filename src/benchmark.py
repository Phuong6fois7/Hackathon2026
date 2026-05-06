
from pathlib import Path
from datetime import datetime, timezone
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "bronze_articles.parquet"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "silver_articles.parquet"
GOLD_SUMMARY_PATH = PROJECT_ROOT / "data" / "gold" / "gold_summaries.parquet"
GOLD_CHUNKS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_chunks.parquet"
GOLD_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_embeddings.npy"
GOLD_EMBEDDING_METADATA_PATH = PROJECT_ROOT / "data" / "gold" / "gold_embedding_metadata.parquet"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "gold" / "faiss_index.bin"

REPORTS_DIR = PROJECT_ROOT / "reports"
BENCHMARK_PATH = REPORTS_DIR / "final_benchmark.csv"


def get_file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def measure_read_time(path: Path) -> Tuple[float, int]:
    """
    Measure how long it takes to read a Parquet file.
    Returns duration in seconds and number of rows.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    start = time.time()
    df = pd.read_parquet(path)
    duration_s = time.time() - start

    return duration_s, len(df)


def add_benchmark_row(
    experiment: str,
    variant: str,
    duration_s: float,
    energy_kwh: Optional[float],
    co2_kg: Optional[float],
    quality_score: Optional[float],
    notes: str,
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "variant": variant,
        "duration_s": duration_s,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "quality_score": quality_score,
        "notes": notes,
    }


def benchmark_parquet_reads() -> pd.DataFrame:
    rows = []

    files_to_test = [
        ("read_bronze", "bronze_parquet", BRONZE_PATH),
        ("read_silver", "silver_parquet", SILVER_PATH),
        ("read_gold_summaries", "gold_summary_parquet", GOLD_SUMMARY_PATH),
        ("read_gold_chunks", "gold_chunks_parquet", GOLD_CHUNKS_PATH),
        ("read_gold_embedding_metadata", "gold_embedding_metadata_parquet", GOLD_EMBEDDING_METADATA_PATH),
    ]

    for experiment, variant, path in files_to_test:
        duration_s, row_count = measure_read_time(path)
        file_size_mb = get_file_size_mb(path)

        rows.append(
            add_benchmark_row(
                experiment=experiment,
                variant=variant,
                duration_s=duration_s,
                energy_kwh=None,
                co2_kg=None,
                quality_score=None,
                notes=f"Rows={row_count}; file_size_mb={file_size_mb:.4f}",
            )
        )

    return pd.DataFrame(rows)


def save_benchmark(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if BENCHMARK_PATH.exists() and BENCHMARK_PATH.stat().st_size > 0:
        existing = pd.read_csv(BENCHMARK_PATH)
        df = pd.concat([existing, df], ignore_index=True)

    df.to_csv(BENCHMARK_PATH, index=False)

    print(f"Benchmark saved to: {BENCHMARK_PATH}")
    print(df.tail())

def benchmark_numpy_file(path: Path) -> dict:
    """
    Benchmark loading a NumPy embeddings file.
    """
    import numpy as np

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    start = time.time()
    embeddings = np.load(path)
    duration_s = time.time() - start

    file_size_mb = get_file_size_mb(path)

    return add_benchmark_row(
        experiment="read_gold_embeddings",
        variant="numpy_npy",
        duration_s=duration_s,
        energy_kwh=None,
        co2_kg=None,
        quality_score=None,
        notes=f"shape={embeddings.shape}; file_size_mb={file_size_mb:.4f}",
    )

def benchmark_faiss_index(path: Path) -> dict:
    """
    Benchmark loading a FAISS index file.
    """
    import faiss

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    start = time.time()

    index = faiss.read_index(str(path))

    duration_s = time.time() - start

    file_size_mb = get_file_size_mb(path)

    return add_benchmark_row(
        experiment="read_faiss_index",
        variant="faiss_binary",
        duration_s=duration_s,
        energy_kwh=None,
        co2_kg=None,
        quality_score=None,
        notes=f"vectors={index.ntotal}; file_size_mb={file_size_mb:.4f}",
    )

def main() -> None:
    print("Running benchmark...")

    benchmark_df = benchmark_parquet_reads()
    embedding_row = benchmark_numpy_file(GOLD_EMBEDDINGS_PATH)

    faiss_row = benchmark_faiss_index(FAISS_INDEX_PATH)

    benchmark_df = pd.concat(
        [
            benchmark_df,
            pd.DataFrame([embedding_row]),
            pd.DataFrame([faiss_row]),
        ],
        ignore_index=True
    )
    save_benchmark(benchmark_df)


if __name__ == "__main__":
    main()