"""
ingestion.py — Polars-based TSV reader for Business Entity Resolution.

Responsibilities:
- Read tab-separated source files with explicit sep="\t"
- dtype=str for all columns (infer_schema_length=0, null_values=[])
- Preserve row order of source1 for output alignment
- Open-set country: no hardcoded list, country kept as raw string
- Expose load_tsv(), load_train_sources(), load_test_sources(),
  load_ground_truth(), build_corpus() helpers
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# ── Column schema expected in every source TSV ───────────────────────────────
SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}

REQUIRED_COLS = list(SCHEMA.keys())

GT_SCHEMA = {
    "source1_entity_id": pl.Utf8,
    "matched_entity_ids": pl.Utf8,
}


def load_tsv(path: str | Path) -> pl.DataFrame:
    """Read one entity source TSV file.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the .tsv file.

    Returns
    -------
    pl.DataFrame
        Columns: entity_id, business_name, business_address, country
        Row order preserved exactly as in file. Nulls filled with "".
        Country is kept as raw string — no normalization, no filtering.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pl.read_csv(
        path,
        separator="\t",
        schema_overrides=SCHEMA,
        null_values=[],          # nothing treated as NaN; addresses have commas
        infer_schema_length=0,   # all cols as Utf8, no type inference
        has_header=True,
        truncate_ragged_lines=True,  # tolerate trailing whitespace / extra cols
        encoding="utf8",
    )

    # Validate expected columns are present
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns {missing}")

    # Select only expected columns (drop any extra) and fill nulls with ""
    df = df.select(REQUIRED_COLS).with_columns(
        [pl.col(c).fill_null("") for c in REQUIRED_COLS]
    )

    return df


# Alias — kept for internal use by other modules
load_source = load_tsv


def load_ground_truth(path: str | Path) -> pl.DataFrame:
    """Read a ground truth TSV file.

    Expected schema: source1_entity_id \\t matched_entity_ids
    matched_entity_ids is a comma-separated string of IDs.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    pl.DataFrame
        Columns: source1_entity_id (Utf8), matched_entity_ids (Utf8)
        Nulls in matched_entity_ids filled with "" (singletons).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    gt = pl.read_csv(
        path,
        separator="\t",
        schema_overrides=GT_SCHEMA,
        null_values=[],
        infer_schema_length=0,
        has_header=True,
        encoding="utf8",
    ).with_columns(pl.col("matched_entity_ids").fill_null(""))

    return gt


def load_train_sources(
    train_dir: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load all train source TSVs plus the ground truth.

    Parameters
    ----------
    train_dir : str or Path
        Directory containing train_source1.tsv, train_source2.tsv,
        train_source3.tsv, train_ground_truth.tsv.

    Returns
    -------
    (s1, s2, s3, gt) — all pl.DataFrame
        gt columns: [source1_entity_id, matched_entity_ids]
    """
    train_dir = Path(train_dir)
    s1 = load_tsv(train_dir / "train_source1.tsv")
    s2 = load_tsv(train_dir / "train_source2.tsv")
    s3 = load_tsv(train_dir / "train_source3.tsv")
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    return s1, s2, s3, gt


def load_test_sources(
    test_dir: str | Path,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load all test source TSVs.

    Parameters
    ----------
    test_dir : str or Path
        Directory containing test_source1.tsv, test_source2.tsv,
        test_source3.tsv.

    Returns
    -------
    (s1, s2, s3) — all pl.DataFrame
        s1 row order is authoritative for output ordering.
    """
    test_dir = Path(test_dir)
    s1 = load_tsv(test_dir / "test_source1.tsv")
    s2 = load_tsv(test_dir / "test_source2.tsv")
    s3 = load_tsv(test_dir / "test_source3.tsv")
    return s1, s2, s3


def build_corpus(s2: pl.DataFrame, s3: pl.DataFrame) -> pl.DataFrame:
    """Concatenate S2 and S3 into the blocking candidate corpus.

    Adds a 'source' column ('S2' or 'S3') for ID disambiguation.
    Row order: all S2 rows first, then all S3 rows.

    Parameters
    ----------
    s2, s3 : pl.DataFrame
        Must both have REQUIRED_COLS.

    Returns
    -------
    pl.DataFrame
        Columns: entity_id, business_name, business_address, country, source
    """
    s2_tagged = s2.with_columns(pl.lit("S2").alias("source"))
    s3_tagged = s3.with_columns(pl.lit("S3").alias("source"))
    return pl.concat([s2_tagged, s3_tagged], how="vertical")
