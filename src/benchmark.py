from pathlib import Path
from datetime import datetime, timezone
import time
from typing import Optional, Tuple

import pandas as pd
from codecarbon import EmissionsTracker


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEAM_NAME = "GreenPubMed_Team"

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "bronze_articles.parquet"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "silver_articles.parquet"

GOLD_SUMMARY_PATH = PROJECT_ROOT / "data" / "gold" / "gold_summaries.parquet"
GOLD_CHUNKS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_chunks.parquet"
GOLD_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "gold" / "gold_embeddings.npy"

REPORTS_DIR = PROJECT_ROOT / "reports"
BENCHMARK_PATH = REPORTS_DIR / "final_benchmark.csv"


def get_file_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def add_benchmark_row(
    experiment_id: str,
    track: str,
    operation: str,
    variant: str,
    sample_size: Optional[int],
    duration_s: float,
    energy_kwh: Optional[float],
    co2_kg: Optional[float],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    estimated_cost: Optional[float],
    model_name: Optional[str],
    prompt_version: Optional[str],
    quality_score: Optional[float],
    notes: str,
) -> dict:
    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "team": TEAM_NAME,
        "track": track,
        "operation": operation,
        "variant": variant,
        "sample_size": sample_size,
        "duration_s": duration_s,
        "energy_kwh": energy_kwh,
        "co2_kg": co2_kg,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": estimated_cost,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "quality_score": quality_score,
        "notes": notes,
    }


def measure_parquet_read(path: Path) -> Tuple[float, int]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    start = time.time()
    df = pd.read_parquet(path)
    duration_s = time.time() - start

    return duration_s, len(df)


def benchmark_parquet_reads() -> pd.DataFrame:
    rows = []

    files_to_test = [
        ("read_bronze_001", "data_engineering", "read_parquet", "bronze_parquet", BRONZE_PATH),
        ("read_silver_001", "data_engineering", "read_parquet", "silver_parquet", SILVER_PATH),
        ("read_gold_summaries_001", "ai", "read_parquet", "gold_summaries_parquet", GOLD_SUMMARY_PATH),
        ("read_gold_chunks_001", "rag", "read_parquet", "gold_chunks_parquet", GOLD_CHUNKS_PATH),
    ]

    for experiment_id, track, operation, variant, path in files_to_test:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        duration_s, row_count = measure_parquet_read(path)
        file_size_mb = get_file_size_mb(path)

        rows.append(
            add_benchmark_row(
                experiment_id=experiment_id,
                track=track,
                operation=operation,
                variant=variant,
                sample_size=row_count,
                duration_s=duration_s,
                energy_kwh=None,
                co2_kg=None,
                input_tokens=None,
                output_tokens=None,
                estimated_cost=None,
                model_name=None,
                prompt_version=None,
                quality_score=None,
                notes=f"file_size_mb={file_size_mb:.4f}",
            )
        )

    return pd.DataFrame(rows)


def benchmark_embeddings_file() -> Optional[dict]:
    if not GOLD_EMBEDDINGS_PATH.exists():
        print(f"Skipping missing embeddings file: {GOLD_EMBEDDINGS_PATH}")
        return None

    import numpy as np

    start = time.time()
    embeddings = np.load(GOLD_EMBEDDINGS_PATH)
    duration_s = time.time() - start

    file_size_mb = get_file_size_mb(GOLD_EMBEDDINGS_PATH)

    return add_benchmark_row(
        experiment_id="read_embeddings_001",
        track="rag",
        operation="read_numpy_embeddings",
        variant="gold_embeddings_npy",
        sample_size=embeddings.shape[0],
        duration_s=duration_s,
        energy_kwh=None,
        co2_kg=None,
        input_tokens=None,
        output_tokens=None,
        estimated_cost=None,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        prompt_version=None,
        quality_score=None,
        notes=f"embedding_dim={embeddings.shape[1]}; file_size_mb={file_size_mb:.4f}",
    )


def benchmark_codecarbon_silver_sample() -> dict:
    """
    First CodeCarbon measurement for Day 1 / section 3.5.

    Measures one local operation:
    read Bronze, compute article length, and write a temporary Silver sample.
    """

    if not BRONZE_PATH.exists():
        raise FileNotFoundError(f"File not found: {BRONZE_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="day1_sample_profile",
        output_dir=str(REPORTS_DIR),
        output_file="emissions.csv",
    )

    tracker.start()
    start = time.time()

    df = pd.read_parquet(BRONZE_PATH)
    df["article_chars"] = df["article"].astype(str).str.len()

    temp_output = PROJECT_ROOT / "data" / "silver" / "sample_articles_codecarbon.parquet"
    df.to_parquet(temp_output, index=False, compression="snappy")

    duration_s = time.time() - start
    emissions = tracker.stop()

    file_size_mb = get_file_size_mb(temp_output)

    return add_benchmark_row(
        experiment_id="codecarbon_silver_sample_001",
        track="green_ai",
        operation="profile_and_write_parquet",
        variant="codecarbon_sample",
        sample_size=len(df),
        duration_s=duration_s,
        energy_kwh=None,
        co2_kg=emissions,
        input_tokens=None,
        output_tokens=None,
        estimated_cost=None,
        model_name=None,
        prompt_version=None,
        quality_score=None,
        notes=f"temporary_file=sample_articles_codecarbon.parquet; file_size_mb={file_size_mb:.4f}; detailed energy in reports/emissions.csv",
    )


def benchmark_storage_formats() -> pd.DataFrame:
    """
    Compare CSV and Parquet storage using the same Silver dataset.
    """

    if not SILVER_PATH.exists():
        raise FileNotFoundError(f"Silver file not found: {SILVER_PATH}")

    df = pd.read_parquet(SILVER_PATH)

    storage_dir = PROJECT_ROOT / "data" / "gold" / "storage_benchmark"
    storage_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("csv_semicolon", storage_dir / "articles.csv"),
        ("parquet_snappy", storage_dir / "articles_snappy.parquet"),
        ("parquet_gzip", storage_dir / "articles_gzip.parquet"),
    ]

    rows = []

    for variant, output_path in variants:
        start = time.time()

        if variant == "csv_semicolon":
            df.to_csv(output_path, index=False, sep=";")
        elif variant == "parquet_snappy":
            df.to_parquet(output_path, index=False, compression="snappy")
        elif variant == "parquet_gzip":
            df.to_parquet(output_path, index=False, compression="gzip")

        write_time = time.time() - start

        start = time.time()

        if variant == "csv_semicolon":
            df_read = pd.read_csv(output_path, sep=";")
        else:
            df_read = pd.read_parquet(output_path)

        read_time = time.time() - start

        file_size_mb = get_file_size_mb(output_path)

        rows.append(
            add_benchmark_row(
                experiment_id=f"storage_{variant}_001",
                track="data_engineering",
                operation="storage_format_benchmark",
                variant=variant,
                sample_size=len(df_read),
                duration_s=write_time + read_time,
                energy_kwh=None,
                co2_kg=None,
                input_tokens=None,
                output_tokens=None,
                estimated_cost=None,
                model_name=None,
                prompt_version=None,
                quality_score=None,
                notes=f"write_time_s={write_time:.4f}; read_time_s={read_time:.4f}; file_size_mb={file_size_mb:.4f}",
            )
        )

    return pd.DataFrame(rows)


def save_benchmark(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    final_columns = [
        "experiment_id",
        "timestamp",
        "team",
        "track",
        "operation",
        "variant",
        "sample_size",
        "duration_s",
        "energy_kwh",
        "co2_kg",
        "input_tokens",
        "output_tokens",
        "estimated_cost",
        "model_name",
        "prompt_version",
        "quality_score",
        "notes",
    ]

    df = df[final_columns]

    # Clean benchmark mode: overwrite previous results
    df.to_csv(BENCHMARK_PATH, index=False)

    print(f"\nBenchmark saved to: {BENCHMARK_PATH}")
    print("\nBenchmark rows:")
    print(df)

def main() -> None:
    print("Running Day 1 benchmark up to section 3.5...")

    benchmark_df = benchmark_parquet_reads()

    embedding_row = benchmark_embeddings_file()
    codecarbon_row = benchmark_codecarbon_silver_sample()

    extra_rows = [codecarbon_row]

    if embedding_row is not None:
        extra_rows.append(embedding_row)

    storage_df = benchmark_storage_formats()

    benchmark_df = pd.concat(
        [benchmark_df, storage_df, pd.DataFrame(extra_rows)],
        ignore_index=True,
    )

    save_benchmark(benchmark_df)


if __name__ == "__main__":
    main()