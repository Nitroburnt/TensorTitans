"""
pipeline.py — End-to-end orchestrator for Business Entity Resolution.

Memory budget (16 GB machine, 10M corpus):
  - Raw TSVs for S2+S3 each ~400 MB when loaded as Polars Utf8.
  - Preprocessed corpus ~2 GB for all 10M rows.
  - Sentence-transformer model ~500 MB weights + mmap.
  - FAISS HNSW index for 10M × 384d float32 = ~15 GB naively — but we shard by
    country, so each shard (US/India/France) is 3-4M rows × 384d = ~5-6 GB peak.
    After build, the index is retained but the raw embeddings are freed.

Memory order in run_train / run_inference:
  1. Load only S1_raw + GT first (small: 2.2M rows ≈ 0.5 GB)
  2. Preprocess S1 only — retain s1 dict, delete s1_raw
  3. Load + preprocess S2/S3 in one go for blocking → delete raw immediately
  4. Initialize BlockingEngine (loads model BEFORE we have the full corpus in RAM)
  5. Build index shard by shard (corpus shard → encode → add to FAISS → del embeddings)
  6. del corpus after indexing; run GC before query
  7. Query → candidates → del engine (frees model + HNSW)
  8. Build pairs_df from s1 dict + corpus_map (lightweight)
  9. del corpus_map, run GC
  10. Feature computation, model training, etc.
"""

from __future__ import annotations

import gc
import pathlib
import pickle
import time
from typing import Optional

import numpy as np
import polars as pl

from .ingestion import load_train_sources, load_test_sources, build_corpus, load_tsv
from .preprocess import preprocess_dataframe
from .blocking import BlockingEngine, evaluate_blocking_recall
from .features import ALL_FEATURE_COLS, compute_features_chunked
from .model import train, predict, save_model, load_model
from .postprocess import postprocess, TAU_DEFAULT

SEED = 42
CHUNK_SIZE = 50_000


def _free(*objs):
    """Delete references and run GC to release RAM."""
    for obj in objs:
        del obj
    gc.collect()


# ── Build pairs DataFrame ──────────────────────────────────────────────────

def build_pairs_df(
    candidates: dict[str, list[str]],
    s1_preprocessed: pl.DataFrame,
    corpus_preprocessed: pl.DataFrame,
    gt: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Build candidate pairs DataFrame from blocking results using Polars joins.

    Uses join-based approach instead of Python dicts to avoid the ~3 GB
    corpus_map overhead on large corpora.

    Returns pl.DataFrame with columns:
      s1_id, cand_id, name1, name2, addr1, addr2, country1, country2, [label]
    """
    # Flatten candidates dict → DataFrame of pairs
    pair_s1: list[str] = []
    pair_cand: list[str] = []
    for s1_id, cand_list in candidates.items():
        for cand_id in cand_list:
            pair_s1.append(s1_id)
            pair_cand.append(cand_id)

    empty_schema: dict = {
        "s1_id": pl.Utf8, "cand_id": pl.Utf8,
        "name1": pl.Utf8, "name2": pl.Utf8,
        "addr1": pl.Utf8, "addr2": pl.Utf8,
        "country1": pl.Utf8, "country2": pl.Utf8,
    }
    if gt is not None:
        empty_schema["label"] = pl.Int32

    if not pair_s1:
        return pl.DataFrame(schema=empty_schema)

    pairs = pl.DataFrame(
        {"s1_id": pair_s1, "cand_id": pair_cand},
        schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8},
    )

    # Extract and rename S1 feature columns for join
    s1_feat = (
        s1_preprocessed
        .select(["entity_id", "name_clean", "address_clean", "country_clean"])
        .rename({"entity_id": "s1_id", "name_clean": "name1",
                 "address_clean": "addr1", "country_clean": "country1"})
    )

    # Extract and rename corpus feature columns for join
    corpus_feat = (
        corpus_preprocessed
        .select(["entity_id", "name_clean", "address_clean", "country_clean"])
        .rename({"entity_id": "cand_id", "name_clean": "name2",
                 "address_clean": "addr2", "country_clean": "country2"})
    )

    result = (
        pairs
        .join(s1_feat, on="s1_id", how="left")
        .join(corpus_feat, on="cand_id", how="left")
    )

    if gt is not None:
        # Build GT label DataFrame for join
        gt_s1: list[str] = []
        gt_cand: list[str] = []
        for row in gt.iter_rows(named=True):
            sid = row["source1_entity_id"]
            matched = row["matched_entity_ids"].strip()
            if matched:
                for cid in matched.split(","):
                    gt_s1.append(sid)
                    gt_cand.append(cid)

        if gt_s1:
            gt_df = pl.DataFrame(
                {"s1_id": gt_s1, "cand_id": gt_cand, "label": [1] * len(gt_s1)},
                schema={"s1_id": pl.Utf8, "cand_id": pl.Utf8, "label": pl.Int32},
            )
            result = result.join(gt_df, on=["s1_id", "cand_id"], how="left")
            result = result.with_columns(pl.col("label").fill_null(0).cast(pl.Int32))
        else:
            result = result.with_columns(pl.lit(0).cast(pl.Int32).alias("label"))

    return result


# ── Memory-efficient corpus loader ────────────────────────────────────────

def _load_and_preprocess_corpus(
    s2_path: pathlib.Path,
    s3_path: pathlib.Path,
) -> pl.DataFrame:
    """Load S2+S3, concatenate, preprocess in one pass, free raw data."""
    from .ingestion import load_tsv, build_corpus as _build_corpus
    s2_raw = load_tsv(s2_path)
    s3_raw = load_tsv(s3_path)
    corpus_raw = _build_corpus(s2_raw, s3_raw)
    del s2_raw, s3_raw
    gc.collect()
    corpus = preprocess_dataframe(corpus_raw)
    del corpus_raw
    gc.collect()
    return corpus


# ── Training pipeline ────────────────────────────────────────────────────────

def run_train(
    train_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    skip_blocking_cache: bool = False,
) -> tuple[list, object, float]:
    """Full training pipeline with memory-efficient staging."""
    train_dir = pathlib.Path(train_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    cand_cache_path = output_dir / "train_candidates.pkl"
    feat_cache_path = output_dir / "train_features.pkl"

    # ── Step 1: Load S1 + GT (small) ─────────────────────────────────────────
    print("\n[pipeline] ── STEP 1: Load S1 + GT ─────────────────────────────────")
    from .ingestion import load_tsv, load_ground_truth
    s1_raw = load_tsv(train_dir / "train_source1.tsv")
    gt     = load_ground_truth(train_dir / "train_ground_truth.tsv")
    print(f"[pipeline] S1={len(s1_raw):,}  GT={len(gt):,}")

    t0 = time.time()
    s1 = preprocess_dataframe(s1_raw)
    del s1_raw; gc.collect()
    print(f"[pipeline] S1 preprocessing done in {time.time()-t0:.1f}s")
    all_s1_ids = s1["entity_id"].to_list()

    # ── Step 2: Blocking (or load cache) ─────────────────────────────────────
    candidates: dict[str, list[str]] = {}

    if not skip_blocking_cache and cand_cache_path.exists():
        print(f"\n[pipeline] Loaded cache: {cand_cache_path}")
        with open(cand_cache_path, "rb") as f:
            candidates = pickle.load(f)
        corpus = None  # will be loaded later for build_pairs_df
        print(f"[pipeline] Cached candidates: {len(candidates):,} S1 entities")
    else:
        print("\n[pipeline] ── STEP 2: Load corpus + Blocking ───────────────────")
        # Load S2+S3 and preprocess corpus
        t0 = time.time()
        corpus = _load_and_preprocess_corpus(
            train_dir / "train_source2.tsv",
            train_dir / "train_source3.tsv",
        )
        print(f"[pipeline] Corpus loaded+preprocessed: {len(corpus):,} rows in {time.time()-t0:.1f}s")

        # Initialize BlockingEngine (loads model) — corpus is in RAM but we're about
        # to encode it shard by shard inside build(); the raw Polars DataFrame is the
        # only large object at this point.
        t0 = time.time()
        engine = BlockingEngine(bm25_only=True)
        engine.build(corpus)
        candidates = engine.query_all(s1, corpus_df=corpus)
        elapsed = time.time() - t0
        print(f"[pipeline] Blocking done in {elapsed:.1f}s  ({len(candidates):,} S1 entities)")

        # Free the engine (BM25 inverted index) — we no longer need it
        del engine; gc.collect()
        print(f"[pipeline] BlockingEngine freed")

        with open(cand_cache_path, "wb") as f:
            pickle.dump(candidates, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[pipeline] Saved blocking cache → {cand_cache_path}")

    # ── Step 3: Evaluate blocking recall ─────────────────────────────────────
    print("\n[pipeline] ── STEP 3: Blocking Recall ──────────────────────────────")
    recall = evaluate_blocking_recall(candidates, gt, label="train")
    if recall < 0.90:
        print(f"[pipeline] WARNING: blocking recall={recall:.4f} < 0.90")
    else:
        print(f"[pipeline] ✓ Blocking recall={recall:.4f} ≥ 0.90")
    del gt; gc.collect()

    # ── Step 4: Build pairs DataFrame (needs corpus for name/addr lookup) ────
    print("\n[pipeline] ── STEP 4: Build Pairs DataFrame ────────────────────────")
    if corpus is None:
        # Load corpus fresh for lookup (candidates were cached, corpus was not kept)
        t0 = time.time()
        corpus = _load_and_preprocess_corpus(
            train_dir / "train_source2.tsv",
            train_dir / "train_source3.tsv",
        )
        print(f"[pipeline] Corpus reloaded for pairs in {time.time()-t0:.1f}s")

    t0 = time.time()
    # Re-load GT for label assignment
    gt = load_ground_truth(train_dir / "train_ground_truth.tsv")
    pairs_df = build_pairs_df(candidates, s1, corpus, gt=gt)
    del corpus, candidates, gt; gc.collect()
    print(f"[pipeline] Pairs: {len(pairs_df):,} rows  "
          f"pos_rate={pairs_df['label'].mean():.4f}  ({time.time()-t0:.1f}s)")

    # ── Step 5: Feature computation ───────────────────────────────────────────
    if not skip_blocking_cache and feat_cache_path.exists():
        del pairs_df; gc.collect()
        print(f"\n[pipeline] Loaded cache: {feat_cache_path}")
        with open(feat_cache_path, "rb") as f:
            feature_df = pickle.load(f)
        print(f"[pipeline] Cached features: {len(feature_df):,} rows  "
              f"{len(feature_df.columns)} cols")
    else:
        print("\n[pipeline] ── STEP 5: Feature Computation ───────────────────────")
        t0 = time.time()
        feature_df = compute_features_chunked(
            pairs_df,
            chunk_size=CHUNK_SIZE,
            name1_col="name1", name2_col="name2",
            addr1_col="addr1", addr2_col="addr2",
            country1_col="country1", country2_col="country2",
        )
        del pairs_df; gc.collect()
        print(f"[pipeline] Features done in {time.time()-t0:.1f}s  "
              f"{len(feature_df):,} rows × {len(ALL_FEATURE_COLS)} feature cols")

        keep_cols = ["s1_id", "cand_id", "label"] + ALL_FEATURE_COLS
        feature_df = feature_df.select(keep_cols)
        with open(feat_cache_path, "wb") as f:
            pickle.dump(feature_df, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[pipeline] Saved feature cache → {feat_cache_path}")

    train_df = feature_df.select(["s1_id", "label"] + ALL_FEATURE_COLS)

    # ── Step 6: Train model ───────────────────────────────────────────────────
    print("\n[pipeline] ── STEP 6: Train LightGBM ──────────────────────────────")
    models, calibrator, fold_scores = train(
        train_df, label_col="label", group_col="s1_id",
    )
    save_model(models, calibrator, output_dir / "model.pkl")
    mean_f05 = float(np.mean(fold_scores))
    print(f"[pipeline] Mean fold F0.5 (pre-calib, thr=0.5): {mean_f05:.4f}")

    # ── Step 7: Find τ* ────────────────────────────────────────────────────────
    print("\n[pipeline] ── STEP 7: Find τ* (threshold sweep) ───────────────────")
    oof_scores = predict(feature_df.select(["s1_id"] + ALL_FEATURE_COLS), models, calibrator)
    scored_for_sweep = feature_df.select(["s1_id", "label"]).with_columns(
        pl.Series("score", oof_scores, dtype=pl.Float32)
    )
    del feature_df; gc.collect()

    from .postprocess import find_tau_star
    tau_star = find_tau_star(scored_for_sweep, label_col="label",
                             score_col="score", group_col="s1_id")

    tau_path = output_dir / "tau_star.txt"
    tau_path.write_text(str(tau_star))
    print(f"[pipeline] τ* = {tau_star:.4f} → saved to {tau_path}")
    print(f"\n[pipeline] ✓ Training complete in {time.time()-t_total:.1f}s")
    return models, calibrator, tau_star


# ── Inference pipeline ───────────────────────────────────────────────────────

def run_inference(
    test_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    train_dir: Optional[str | pathlib.Path] = None,
) -> None:
    """Full inference pipeline with memory-efficient staging."""
    test_dir = pathlib.Path(test_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ── Step 1: Load model ───────────────────────────────────────────────────
    model_path = output_dir / "model.pkl"
    tau_path   = output_dir / "tau_star.txt"

    if not model_path.exists():
        if train_dir is not None:
            print(f"\n[pipeline] model.pkl not found. Auto-training from {train_dir}...")
            models, calibrator, tau_star = run_train(train_dir, output_dir)
        else:
            raise RuntimeError(
                f"model.pkl not found at {model_path}. "
                "Provide --train_dir to auto-train, or run --train_only first."
            )
    else:
        print("\n[pipeline] ── STEP 1: Load Model ─────────────────────────────")
        models, calibrator = load_model(model_path)
        tau_star = float(tau_path.read_text().strip()) if tau_path.exists() else TAU_DEFAULT
        print(f"[pipeline] τ* = {tau_star:.4f}")

    # ── Step 2: Load S1 only first (preserve s1_raw for final all_s1_ids) ────
    print("\n[pipeline] ── STEP 2: Load + Preprocess Test Data ─────────────────")
    from .ingestion import load_tsv
    t0 = time.time()
    s1_raw = load_tsv(test_dir / "test_source1.tsv")
    all_s1_ids = s1_raw["entity_id"].to_list()
    s1 = preprocess_dataframe(s1_raw)
    del s1_raw; gc.collect()
    print(f"[pipeline] S1={len(s1):,}  preprocessed in {time.time()-t0:.1f}s")

    # ── Step 3: Load corpus + Blocking ───────────────────────────────────────
    print("\n[pipeline] ── STEP 3: Load Corpus + Blocking ───────────────────────")
    t0 = time.time()
    corpus = _load_and_preprocess_corpus(
        test_dir / "test_source2.tsv",
        test_dir / "test_source3.tsv",
    )
    print(f"[pipeline] Corpus: {len(corpus):,} rows in {time.time()-t0:.1f}s")

    t0 = time.time()
    engine = BlockingEngine(bm25_only=True)
    engine.build(corpus)
    candidates = engine.query_all(s1, corpus_df=corpus)
    del engine; gc.collect()
    elapsed = time.time() - t0
    avg_cands = sum(len(v) for v in candidates.values()) / max(len(candidates), 1)
    print(f"[pipeline] Blocking done in {elapsed:.1f}s  "
          f"({len(candidates):,} S1 entities, avg cands/S1: {avg_cands:.1f})")

    # ── Step 4: Build pairs DataFrame ────────────────────────────────────────
    print("\n[pipeline] ── STEP 4: Build Pairs DataFrame ────────────────────────")
    t0 = time.time()
    pairs_df = build_pairs_df(candidates, s1, corpus, gt=None)
    del corpus, candidates; gc.collect()
    print(f"[pipeline] Pairs: {len(pairs_df):,} rows  ({time.time()-t0:.1f}s)")

    # ── Step 5: Feature computation ───────────────────────────────────────────
    print("\n[pipeline] ── STEP 5: Feature Computation ──────────────────────────")
    t0 = time.time()
    feature_df = compute_features_chunked(
        pairs_df,
        chunk_size=CHUNK_SIZE,
        name1_col="name1", name2_col="name2",
        addr1_col="addr1", addr2_col="addr2",
        country1_col="country1", country2_col="country2",
    )
    del pairs_df; gc.collect()
    print(f"[pipeline] Features done in {time.time()-t0:.1f}s  "
          f"{len(feature_df):,} rows × {len(ALL_FEATURE_COLS)} feature cols")

    # ── Step 6: Predict ────────────────────────────────────────────────────
    print("\n[pipeline] ── STEP 6: Predict ───────────────────────────────────────")
    t0 = time.time()
    scores = predict(feature_df, models, calibrator)
    del models, calibrator; gc.collect()
    print(f"[pipeline] Prediction done in {time.time()-t0:.1f}s  "
          f"score range=[{scores.min():.4f}, {scores.max():.4f}]")

    scored_df = feature_df.select(["s1_id", "cand_id"]).with_columns(
        pl.Series("score", scores, dtype=pl.Float32)
    )
    del feature_df; gc.collect()

    # ── Step 7: Postprocess → write outputs ──────────────────────────────────
    print("\n[pipeline] ── STEP 7: Postprocess ──────────────────────────────────")
    predictions, tau_used = postprocess(
        scored_df=scored_df,
        all_s1_ids=all_s1_ids,
        output_dir=output_dir,
        tau=tau_star,
        label_col=None,
        score_col="score",
        group_col="s1_id",
        cand_id_col="cand_id",
    )

    n_matched   = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    print(f"\n[pipeline] ✓ Inference complete in {time.time()-t_total:.1f}s")
    print(f"[pipeline] Output: {n_matched} matched, {n_singleton} singletons")
    print(f"[pipeline]   → {output_dir / 'matching_results.tsv'}")
    print(f"[pipeline]   → {output_dir / 'candidate_pairs.tsv'}")
