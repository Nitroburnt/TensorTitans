"""
model.py — LightGBM ranking model for Business Entity Resolution.

Architecture (FR-05, SRS v2 §4.3):
  - Binary LightGBM with FP penalty w0=2.0 (aligns with F0.5, precision 2× recall)
  - GroupKFold by S1 entity_id (5 folds) — no data leakage across S1 groups
  - Singleton preservation: groups that are all-negative are distributed evenly
    across folds so each fold sees representative singleton proportion
  - Isotonic calibration fitted on OOF (out-of-fold) predictions
  - Macro-averaged F0.5 evaluation metric (strict singleton rule)

Exports:
  SEED, N_FOLDS, W0, LGBM_PARAMS       constants
  make_folds(df, ...)                   GroupKFold with singleton preservation
  macro_f05(df, ...)                    macro-averaged F0.5 scorer
  train(feature_df, ...)                → (models, calibrator, fold_scores)
  predict(feature_df, models, calib)    → calibrated probability array
  save_model(models, calib, path)       pickle
  load_model(path)                      unpickle
"""

from __future__ import annotations

import argparse
import pickle
import pathlib
import time
from typing import Optional

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from .features import ALL_FEATURE_COLS, compute_all_features, compute_features_chunked

# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42
N_FOLDS = 5
W0 = 2.0          # FP class weight (class 0 = non-match)

LGBM_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}


# ── F0.5 metric ───────────────────────────────────────────────────────────────

def macro_f05(
    df: pl.DataFrame,
    pred_col: str = "pred",
    label_col: str = "label",
    group_col: str = "s1_id",
    threshold: float = 0.5,
) -> float:
    """Compute macro-averaged F0.5 over S1 groups.

    For each S1 group:
      - positive predictions = rows where pred >= threshold
      - precision = TP / (TP + FP)  [1.0 if no positives predicted]
      - recall    = TP / (TP + FN)  [1.0 if no GT positives]
      - F0.5      = 1.25 * P * R / (0.25 * P + R)  [0.0 if P+R==0]

    Singleton rule:
      - label.sum() == 0 AND pred_positive.sum() == 0  →  F0.5 = 1.0
      - label.sum() == 0 AND pred_positive.sum() > 0   →  F0.5 = 0.0 (FP on singleton)

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: group_col (str), label_col (int/float 0/1),
        pred_col (float probability).
    pred_col : str
        Column of predicted probabilities.
    label_col : str
        Binary ground-truth label column.
    group_col : str
        S1 entity ID column for grouping.
    threshold : float
        Decision threshold for positive prediction.

    Returns
    -------
    float
        Macro-averaged F0.5 across all S1 groups.
    """
    # Convert to numpy for speed
    groups = df[group_col].to_numpy()
    labels = df[label_col].to_numpy().astype(np.float32)
    preds  = df[pred_col].to_numpy().astype(np.float32)
    positives = (preds >= threshold).astype(np.float32)

    unique_groups = np.unique(groups)
    f05_scores: list[float] = []

    for gid in unique_groups:
        mask = groups == gid
        y = labels[mask]
        yp = positives[mask]

        gt_sum  = y.sum()
        pp_sum  = yp.sum()

        # Singleton rule
        if gt_sum == 0:
            f05_scores.append(1.0 if pp_sum == 0 else 0.0)
            continue

        tp = float((y * yp).sum())
        fp = float(((1 - y) * yp).sum())
        fn = float((y * (1 - yp)).sum())

        denom_p = tp + fp
        precision = tp / denom_p if denom_p > 0 else 1.0

        denom_r = tp + fn
        recall = tp / denom_r if denom_r > 0 else 1.0

        denom_f = 0.25 * precision + recall
        f05 = (1.25 * precision * recall / denom_f) if denom_f > 0 else 0.0
        f05_scores.append(f05)

    return float(np.mean(f05_scores)) if f05_scores else 0.0


# ── GroupKFold with singleton preservation ────────────────────────────────────

def make_folds(
    df: pl.DataFrame,
    label_col: str = "label",
    group_col: str = "s1_id",
    n_folds: int = N_FOLDS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create GroupKFold splits preserving singleton proportion per fold.

    All pairs for the same S1 entity stay in the same fold (no leakage).
    Singleton groups (all-negative S1 entities) are distributed evenly across
    folds by assigning them round-robin after sorting by group ID (deterministic).

    Parameters
    ----------
    df : pl.DataFrame
        Must have columns: group_col, label_col.
    label_col : str
        Binary label (1=match, 0=non-match).
    group_col : str
        S1 entity ID column.
    n_folds : int
        Number of cross-validation folds.

    Returns
    -------
    list of (train_indices, val_indices)
        Each element is a tuple of int64 numpy arrays (row indices into df).
    """
    idx = np.arange(len(df), dtype=np.int64)
    groups_arr = df[group_col].to_numpy()
    labels_arr = df[label_col].to_numpy().astype(np.int32)

    # Identify all unique groups and split into matched vs singleton
    unique_groups = np.unique(groups_arr)

    # For each group compute sum of labels
    matched_groups: list = []
    singleton_groups: list = []

    for gid in unique_groups:
        mask = groups_arr == gid
        if labels_arr[mask].sum() > 0:
            matched_groups.append(gid)
        else:
            singleton_groups.append(gid)

    # Assign matched groups using GroupKFold (sklearn)
    # Build a reduced frame: one row per group for sklearn's GroupKFold
    matched_groups = sorted(matched_groups)
    singleton_groups = sorted(singleton_groups)

    # Map group → fold index
    group_fold: dict = {}

    if matched_groups:
        # We use GroupKFold on the original df restricted to matched groups
        matched_mask = np.isin(groups_arr, matched_groups)
        matched_idx = idx[matched_mask]
        matched_g   = groups_arr[matched_mask]
        # Dummy X and y for sklearn GroupKFold (only groups matter)
        dummy_X = np.zeros((len(matched_idx), 1), dtype=np.float32)
        dummy_y = np.zeros(len(matched_idx), dtype=np.int32)

        gkf = GroupKFold(n_splits=n_folds)
        # We need per-group fold assignment; iterate and track which fold each group ends up in
        for fold_idx, (_, val_i) in enumerate(gkf.split(dummy_X, dummy_y, groups=matched_g)):
            for g in np.unique(matched_g[val_i]):
                group_fold[g] = fold_idx

    # Distribute singletons evenly (round-robin by sorted group ID)
    # This ensures each fold sees ~same singleton proportion
    rng = np.random.RandomState(SEED)
    shuffled_singletons = list(singleton_groups)
    rng.shuffle(shuffled_singletons)
    for i, gid in enumerate(shuffled_singletons):
        group_fold[gid] = i % n_folds

    # Build per-row fold assignment
    row_fold = np.array([group_fold[g] for g in groups_arr], dtype=np.int32)

    # Build (train_idx, val_idx) pairs
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_i in range(n_folds):
        val_mask   = row_fold == fold_i
        train_mask = ~val_mask
        folds.append((idx[train_mask], idx[val_mask]))

    return folds


# ── Training function ─────────────────────────────────────────────────────────

def train(
    feature_df: pl.DataFrame,
    label_col: str = "label",
    group_col: str = "s1_id",
    feature_cols: Optional[list[str]] = None,
) -> tuple[list, IsotonicRegression, list[float]]:
    """Train LightGBM with GroupKFold + isotonic calibration.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Must have columns: s1_id, label, + all 28 feature columns (or subset).
    label_col : str
        Binary label column (1=match, 0=non-match).
    group_col : str
        S1 entity ID column for GroupKFold grouping.
    feature_cols : list[str] | None
        Which feature columns to use. Defaults to ALL_FEATURE_COLS.

    Returns
    -------
    models : list of trained LGBMClassifier (one per fold)
    calibrator : IsotonicRegression fitted on OOF predictions
    fold_scores : list of per-fold validation F0.5 scores
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS

    # Validate all required columns present
    missing = [c for c in feature_cols if c not in feature_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:5]} ...")

    # Convert to numpy once (avoid repeated conversion per fold)
    X = feature_df.select(feature_cols).to_numpy().astype(np.float32)
    y = feature_df[label_col].to_numpy().astype(np.int32)

    # OOF predictions accumulated here
    oof_preds  = np.zeros(len(feature_df), dtype=np.float64)
    oof_filled = np.zeros(len(feature_df), dtype=bool)

    models: list[lgb.LGBMClassifier] = []
    fold_scores: list[float] = []

    folds = make_folds(feature_df, label_col=label_col, group_col=group_col)

    print(f"[model] Training LightGBM {N_FOLDS}-fold, n={len(feature_df):,}, "
          f"features={len(feature_cols)}, w0={W0}")
    print(f"[model] Positive rate: {y.mean():.4f}  "
          f"(singletons all-neg: "
          f"{(feature_df.group_by(group_col).agg(pl.col(label_col).sum().alias('s')).filter(pl.col('s')==0).height)} "
          f"/ {feature_df[group_col].n_unique()} groups)")

    t_total = time.time()

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx],   y[val_idx]

        # FP penalty: class 0 (non-match) gets W0=2.0
        sample_weight = np.where(y_tr == 0, W0, 1.0).astype(np.float32)

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        # OOF probabilities
        val_probs = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx]  = val_probs
        oof_filled[val_idx] = True

        # Compute per-fold F0.5 at threshold 0.5 (pre-calibration estimate)
        val_rows = feature_df[val_idx.tolist()]
        val_with_pred = val_rows.with_columns(
            pl.Series("pred", val_probs, dtype=pl.Float32)
        )
        f05 = macro_f05(val_with_pred, pred_col="pred", label_col=label_col,
                        group_col=group_col, threshold=0.5)
        fold_scores.append(f05)

        best_iter = model.best_iteration_ if model.best_iteration_ is not None else LGBM_PARAMS["n_estimators"]
        elapsed = time.time() - t0
        print(f"  Fold {fold_i+1}/{N_FOLDS}  best_iter={best_iter:4d}  "
              f"val_size={len(val_idx):,}  F0.5@0.5={f05:.4f}  ({elapsed:.1f}s)")

        models.append(model)

    # Sanity check: all rows should have OOF predictions
    if not oof_filled.all():
        n_missing = (~oof_filled).sum()
        print(f"[model] WARNING: {n_missing} rows have no OOF prediction — filling with 0.5")
        oof_preds[~oof_filled] = 0.5

    # Isotonic calibration: map raw OOF probabilities → calibrated probabilities
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(oof_preds, y)

    mean_f05 = float(np.mean(fold_scores))
    total_t  = time.time() - t_total
    print(f"\n[model] Mean fold F0.5 (pre-calib, thr=0.5): {mean_f05:.4f}  "
          f"(total {total_t:.1f}s)")
    print(f"[model] Isotonic calibration fitted on {len(oof_preds):,} OOF predictions")

    return models, calibrator, fold_scores


# ── Prediction function ───────────────────────────────────────────────────────

def predict(
    feature_df: pl.DataFrame,
    models: list,
    calibrator: IsotonicRegression,
    feature_cols: Optional[list[str]] = None,
) -> np.ndarray:
    """Ensemble + isotonic calibrated prediction.

    Averages predictions across all fold models, then applies isotonic
    calibration.

    Parameters
    ----------
    feature_df : pl.DataFrame
        Must have columns matching feature_cols.
    models : list of LGBMClassifier
        Fold models returned by train().
    calibrator : IsotonicRegression
        Calibrator returned by train().
    feature_cols : list[str] | None
        Defaults to ALL_FEATURE_COLS.

    Returns
    -------
    np.ndarray of shape (N,), dtype float32
        Calibrated match probability for each pair.
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURE_COLS

    X = feature_df.select(feature_cols).to_numpy().astype(np.float32)

    # Average raw probabilities across fold models
    probs = np.zeros(len(X), dtype=np.float64)
    for m in models:
        probs += m.predict_proba(X)[:, 1]
    probs /= len(models)

    # Apply isotonic calibration
    calibrated = calibrator.transform(probs).astype(np.float32)
    return calibrated


# ── Save / load utilities ─────────────────────────────────────────────────────

def save_model(
    models: list,
    calibrator: IsotonicRegression,
    path: str | pathlib.Path,
) -> None:
    """Pickle (models, calibrator) to path.

    Parameters
    ----------
    models : list of LGBMClassifier
    calibrator : IsotonicRegression
    path : str or Path
        Output file path.
    """
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((models, calibrator), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[model] Saved {len(models)} models + calibrator → {path}")


def load_model(
    path: str | pathlib.Path,
) -> tuple[list, IsotonicRegression]:
    """Load and return (models, calibrator) from pickle.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    (list of LGBMClassifier, IsotonicRegression)
    """
    with open(path, "rb") as f:
        models, calibrator = pickle.load(f)
    print(f"[model] Loaded {len(models)} models + calibrator from {path}")
    return models, calibrator


# ── Smoke test / __main__ block ───────────────────────────────────────────────

def _build_train_features(train_dir: pathlib.Path, output_dir: pathlib.Path) -> pl.DataFrame:
    """Build training features from scratch using blocking + feature engineering.

    Returns a DataFrame with columns: s1_id, label, + ALL_FEATURE_COLS.
    """
    from .ingestion import load_train_sources, build_corpus
    from .preprocess import preprocess_dataframe
    from .blocking import BlockingEngine

    print("[smoke] Loading training data...")
    s1_raw, s2_raw, s3_raw, gt = load_train_sources(train_dir)
    print(f"[smoke] S1={len(s1_raw):,}  S2={len(s2_raw):,}  S3={len(s3_raw):,}  GT={len(gt):,}")

    print("[smoke] Preprocessing...")
    s1 = preprocess_dataframe(s1_raw)
    s2 = preprocess_dataframe(s2_raw)
    s3 = preprocess_dataframe(s3_raw)
    corpus = preprocess_dataframe(build_corpus(s2_raw, s3_raw))

    print("[smoke] Building blocking index + candidate pairs...")
    engine = BlockingEngine()
    engine.build(corpus)
    candidates = engine.query_all(s1)

    # Build pairs DataFrame
    print("[smoke] Building pair table...")
    rows = []
    gt_lookup: dict[str, set[str]] = {}
    for row in gt.iter_rows(named=True):
        sid = row["source1_entity_id"]
        matched = row["matched_entity_ids"].strip()
        gt_lookup[sid] = set(matched.split(",")) if matched else set()

    # Merge preprocess results for lookup
    s1_map: dict[str, dict] = {
        r["entity_id"]: r for r in s1.iter_rows(named=True)
    }
    corpus_map: dict[str, dict] = {
        r["entity_id"]: r for r in corpus.iter_rows(named=True)
    }

    for s1_id, cands in candidates.items():
        if s1_id not in s1_map:
            continue
        s1_row = s1_map[s1_id]
        gt_set = gt_lookup.get(s1_id, set())
        for cand_id in cands:
            if cand_id not in corpus_map:
                continue
            c_row = corpus_map[cand_id]
            label = 1 if cand_id in gt_set else 0
            rows.append({
                "s1_id":    s1_id,
                "cand_id":  cand_id,
                "label":    label,
                "name1":    s1_row["name_clean"],
                "name2":    c_row["name_clean"],
                "addr1":    s1_row["address_clean"],
                "addr2":    c_row["address_clean"],
                "country1": s1_row["country_clean"],
                "country2": c_row["country_clean"],
            })

    pairs_df = pl.DataFrame(rows, schema_overrides={
        "label": pl.Int32,
        "s1_id": pl.Utf8, "cand_id": pl.Utf8,
        "name1": pl.Utf8, "name2": pl.Utf8,
        "addr1": pl.Utf8, "addr2": pl.Utf8,
        "country1": pl.Utf8, "country2": pl.Utf8,
    })
    print(f"[smoke] Pair table: {len(pairs_df):,} rows  "
          f"positive rate={pairs_df['label'].mean():.4f}")

    print("[smoke] Computing features (chunked)...")
    t0 = time.time()
    feature_df = compute_features_chunked(pairs_df)
    print(f"[smoke] Features done in {time.time()-t0:.1f}s")

    # Keep only needed columns
    keep_cols = ["s1_id", "label"] + ALL_FEATURE_COLS
    feature_df = feature_df.select(keep_cols)

    # Cache
    cache_path = output_dir / "train_features.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(feature_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[smoke] Cached features → {cache_path}")

    return feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="model.py smoke test / training driver")
    parser.add_argument(
        "--skip-blocking",
        action="store_true",
        help="Load cached train_features.pkl instead of re-running blocking",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(pathlib.Path(__file__).parents[4] / "dataset" / "train"),
        help="Path to train data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(pathlib.Path(__file__).parents[4] / "output"),
        help="Path to output directory",
    )
    args = parser.parse_args()

    train_dir  = pathlib.Path(args.train_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / "train_features.pkl"

    # ── Load or build features ────────────────────────────────────────────────
    if args.skip_blocking and cache_path.exists():
        print(f"[smoke] Loading cached features from {cache_path}")
        with open(cache_path, "rb") as f:
            feature_df = pickle.load(f)
        print(f"[smoke] Loaded {len(feature_df):,} rows, "
              f"{len(feature_df.columns)} cols")
    else:
        feature_df = _build_train_features(train_dir, output_dir)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n[smoke] Starting training...")
    models, calibrator, fold_scores = train(feature_df)

    print(f"\n[smoke] Per-fold F0.5: {[f'{s:.4f}' for s in fold_scores]}")
    mean_f05 = float(np.mean(fold_scores))
    print(f"[smoke] Mean F0.5 (pre-calib, thr=0.5): {mean_f05:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = output_dir / "model.pkl"
    save_model(models, calibrator, model_path)

    # ── Validate mean F0.5 ────────────────────────────────────────────────────
    assert mean_f05 >= 0.70, (
        f"Mean fold F0.5={mean_f05:.4f} below minimum 0.70. "
        "Check features, blocking recall, or class weights."
    )
    print(f"\n[smoke] ✓ mean F0.5={mean_f05:.4f} ≥ 0.70 threshold")
    print("[smoke] model.py smoke test PASSED ✓")
