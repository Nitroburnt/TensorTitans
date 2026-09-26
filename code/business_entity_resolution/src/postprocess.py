"""
postprocess.py — Threshold sweep + singleton gate → output TSV files.

Architecture (task 4.2):
  - τ* sweep: evaluate macro F0.5 at every τ ∈ [0.40, 0.95] step 0.01
    to find the threshold that maximizes precision-biased F0.5.
  - Singleton gate: for each S1 group, if max(score) < τ → predict []
    (critical for precision: avoids false positives on singleton entities)
  - Writes two output files:
      matching_results.tsv  — final predictions (scored on leaderboard)
      candidate_pairs.tsv   — blocking superset K≤30 (all scored pairs)

Singleton gate rationale:
  F0.5 weights precision 2× recall. A false positive on a true singleton
  scores 0.0. Predicting empty on a singleton scores 1.0. So when the model
  is uncertain (all scores below τ) we predict empty rather than guess.

Output TSV headers (exact, as required by validate_submission.py):
  matching_results.tsv  : source1_entity_id\\tmatched_entity_ids
  candidate_pairs.tsv   : source1_entity_id\\tcandidate_entity_ids

Exports:
  TAU_MIN, TAU_MAX, TAU_STEP, TAU_DEFAULT   constants
  sweep_threshold(scored_df, ...)           → (tau_star, curve)
  apply_threshold(scored_df, tau, ...)      → dict[s1_id, list[cand_id]]
  find_tau_star(scored_df, ...)             → float
  write_matching_results(predictions, ...)  → None (writes TSV)
  write_candidate_pairs(scored_df, ...)     → None (writes TSV)
  postprocess(scored_df, ...)               → (predictions, tau_used)
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import polars as pl

from .model import macro_f05

# ── Constants ─────────────────────────────────────────────────────────────────

TAU_MIN: float = 0.40
TAU_MAX: float = 0.95
TAU_STEP: float = 0.01
TAU_DEFAULT: float = 0.78   # expected peak τ* for F0.5 (used when no GT available)


# ── Threshold sweep ───────────────────────────────────────────────────────────

def sweep_threshold(
    scored_df: pl.DataFrame,
    label_col: str = "label",
    score_col: str = "score",
    group_col: str = "s1_id",
    tau_min: float = TAU_MIN,
    tau_max: float = TAU_MAX,
    tau_step: float = TAU_STEP,
) -> tuple[float, list[tuple[float, float]]]:
    """Sweep τ from tau_min to tau_max (inclusive, step tau_step).

    For each τ:
      - Apply singleton gate: if max(score) < τ for a group → predict []
      - Compute macro_f05(df, threshold=τ)

    The singleton gate is baked into macro_f05 via threshold: when all scores in
    a group fall below τ, no positives are predicted, so the group is treated as
    a singleton prediction (empty). macro_f05 handles the singleton rule correctly:
    if GT is all-zero and prediction is empty → F0.5 = 1.0.

    Parameters
    ----------
    scored_df : pl.DataFrame
        Must contain columns: group_col (str), label_col (int/float 0/1),
        score_col (float probability).
    label_col : str
        Binary ground-truth label column.
    score_col : str
        Column of predicted probabilities.
    group_col : str
        S1 entity ID column for grouping.
    tau_min, tau_max, tau_step : float
        Sweep range (inclusive on both ends).

    Returns
    -------
    tau_star : float
        τ with the highest macro F0.5.
    curve : list of (tau, f05) pairs
        Full sweep curve for analysis/plotting.
    """
    # Build tau grid: use round() to avoid float precision drift
    n_steps = round((tau_max - tau_min) / tau_step) + 1
    taus = [round(tau_min + i * tau_step, 10) for i in range(n_steps)]

    # Rename score_col to "pred" so macro_f05 pred_col matches
    # (macro_f05 accepts pred_col param, so pass it directly)
    curve: list[tuple[float, float]] = []
    best_tau = TAU_DEFAULT
    best_f05 = -1.0

    for tau in taus:
        f05 = macro_f05(
            scored_df,
            pred_col=score_col,
            label_col=label_col,
            group_col=group_col,
            threshold=tau,
        )
        curve.append((tau, f05))
        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau

    return best_tau, curve


# ── Apply threshold (inference — no GT needed) ────────────────────────────────

def apply_threshold(
    scored_df: pl.DataFrame,
    tau: float,
    score_col: str = "score",
    group_col: str = "s1_id",
    cand_id_col: str = "cand_id",
) -> dict[str, list[str]]:
    """Apply threshold τ and singleton gate to produce final predictions.

    For each S1 group:
      - Singleton gate: if max(score) < τ → predict [] (no matches)
      - Otherwise: all candidates where score >= τ

    Parameters
    ----------
    scored_df : pl.DataFrame
        Must have columns: group_col, cand_id_col, score_col.
    tau : float
        Decision threshold.
    score_col : str
        Column of predicted probabilities.
    group_col : str
        S1 entity ID column.
    cand_id_col : str
        Candidate entity ID column.

    Returns
    -------
    dict[str, list[str]]
        s1_id → list of matched candidate entity IDs (empty list = singleton).
    """
    # Convert to numpy for fast group-wise processing
    s1_ids  = scored_df[group_col].to_numpy()
    cands   = scored_df[cand_id_col].to_numpy()
    scores  = scored_df[score_col].to_numpy().astype(np.float64)

    # Group rows by s1_id
    unique_groups = np.unique(s1_ids)
    predictions: dict[str, list[str]] = {}

    for gid in unique_groups:
        mask = s1_ids == gid
        g_scores = scores[mask]
        g_cands  = cands[mask]

        p_max = g_scores.max() if len(g_scores) > 0 else 0.0

        # Singleton gate
        if p_max < tau:
            predictions[gid] = []
        else:
            # All candidates above threshold
            above = g_cands[g_scores >= tau]
            predictions[gid] = above.tolist()

    return predictions


# ── Write output files ────────────────────────────────────────────────────────

def write_matching_results(
    predictions: dict[str, list[str]],
    all_s1_ids: list[str],
    output_path: str | pathlib.Path,
) -> None:
    """Write matching_results.tsv.

    Parameters
    ----------
    predictions : dict[str, list[str]]
        s1_id → list of matched candidate IDs (empty list for singletons).
    all_s1_ids : list[str]
        ALL S1 entity IDs that must appear in the output (even if not in predictions).
    output_path : str or Path
        Destination file path.

    Format
    ------
    Header: source1_entity_id\\tmatched_entity_ids
    - matched_entity_ids is comma-joined list, empty string if no matches
    - Every S1 id in all_s1_ids appears exactly once
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Deduplicate all_s1_ids while preserving order (seen first wins)
    seen_ids: set[str] = set()
    ordered_ids: list[str] = []
    for sid in all_s1_ids:
        if sid not in seen_ids:
            seen_ids.add(sid)
            ordered_ids.append(sid)

    rows = []
    for sid in ordered_ids:
        matched = predictions.get(sid, [])
        rows.append({
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(matched),
        })

    # Write manually to avoid polars quoting empty strings as ""
    n_nonempty = sum(1 for r in rows if r["matched_entity_ids"])
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for r in rows:
            f.write(f"{r['source1_entity_id']}\t{r['matched_entity_ids']}\n")
    print(f"[postprocess] Written {len(rows)} rows to {output_path} "
          f"({n_nonempty} non-empty, {len(rows) - n_nonempty} singletons)")


def write_candidate_pairs(
    scored_df: pl.DataFrame,
    output_path: str | pathlib.Path,
    score_col: str = "score",
    group_col: str = "s1_id",
    cand_id_col: str = "cand_id",
) -> None:
    """Write candidate_pairs.tsv from the scored pairs DataFrame.

    Writes ALL pairs in scored_df (no threshold — this is the blocking superset K≤30).

    Parameters
    ----------
    scored_df : pl.DataFrame
        Must have at least group_col and cand_id_col columns.
    output_path : str or Path
        Destination file path.

    Format
    ------
    Header: source1_entity_id\\tcandidate_entity_ids
    - candidate_entity_ids is comma-joined list of all candidates per S1
    - Each S1 appears exactly once (all its candidates on one row)
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group candidates per S1 (preserving original order within each group)
    s1_ids  = scored_df[group_col].to_numpy()
    cands   = scored_df[cand_id_col].to_numpy()

    # Use insertion-order dict to preserve S1 ordering
    groups: dict[str, list[str]] = {}
    for s1, cand in zip(s1_ids, cands):
        if s1 not in groups:
            groups[s1] = []
        groups[s1].append(cand)

    rows = [
        {
            "source1_entity_id": s1,
            "candidate_entity_ids": ",".join(cand_list),
        }
        for s1, cand_list in groups.items()
    ]

    # Write manually to keep consistent format (avoid polars quoting edge cases)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for r in rows:
            f.write(f"{r['source1_entity_id']}\t{r['candidate_entity_ids']}\n")
    print(f"[postprocess] Written {len(rows)} rows ({len(scored_df)} pairs) "
          f"to {output_path}")


# ── Find τ* on validation set ─────────────────────────────────────────────────

def find_tau_star(
    scored_df: pl.DataFrame,
    label_col: str = "label",
    score_col: str = "score",
    group_col: str = "s1_id",
) -> float:
    """Find τ* that maximizes macro F0.5 on a labeled validation set.

    Prints the sweep curve (tau, f05) for τ in [TAU_MIN, TAU_MAX].
    Prints the best τ* and its F0.5 score.

    Parameters
    ----------
    scored_df : pl.DataFrame
        Must have group_col, label_col, score_col.
    label_col : str
        Binary ground-truth label column.
    score_col : str
        Column of predicted probabilities.
    group_col : str
        S1 entity ID column.

    Returns
    -------
    float
        tau_star — the threshold with the highest macro F0.5.
    """
    tau_star, curve = sweep_threshold(
        scored_df,
        label_col=label_col,
        score_col=score_col,
        group_col=group_col,
    )

    # Print sweep curve
    print(f"\n[postprocess] τ sweep ({len(curve)} steps):")
    print(f"  {'tau':>6}  {'F0.5':>8}")
    for tau, f05 in curve:
        marker = " ← τ*" if tau == tau_star else ""
        print(f"  {tau:6.2f}  {f05:8.4f}{marker}")

    best_f05 = next(f05 for t, f05 in curve if t == tau_star)
    print(f"\n[postprocess] τ* = {tau_star:.2f}  (F0.5 = {best_f05:.4f})")

    return tau_star


# ── End-to-end postprocess ────────────────────────────────────────────────────

def postprocess(
    scored_df: pl.DataFrame,
    all_s1_ids: list[str],
    output_dir: str | pathlib.Path,
    tau: Optional[float] = None,
    label_col: Optional[str] = None,
    score_col: str = "score",
    group_col: str = "s1_id",
    cand_id_col: str = "cand_id",
) -> tuple[dict[str, list[str]], float]:
    """Full postprocessing: threshold → write outputs.

    Parameters
    ----------
    scored_df : pl.DataFrame
        Candidate pairs with score column. Must have group_col, cand_id_col, score_col.
        Optionally has label_col (if tau is None → sweep is used to find tau*).
    all_s1_ids : list[str]
        All S1 entity IDs to include in matching_results.tsv (even singletons).
    output_dir : str or Path
        Directory to write candidate_pairs.tsv and matching_results.tsv.
    tau : float | None
        - None + label_col present → sweep to find tau*.
        - None + no label_col     → use TAU_DEFAULT=0.78.
        - float                   → use this threshold directly.
    label_col : str | None
        If provided and tau is None: run threshold sweep to find tau*.
    score_col : str
        Column name for model scores.
    group_col : str
        S1 entity ID column.
    cand_id_col : str
        Candidate entity ID column.

    Returns
    -------
    predictions : dict[str, list[str]]
        s1_id → list of matched candidate IDs (empty list = singleton).
    tau_used : float
        The threshold that was actually used.

    Side effects
    ------------
    Writes to output_dir:
      - candidate_pairs.tsv   (blocking superset, all scored pairs)
      - matching_results.tsv  (final predictions after threshold + singleton gate)
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine threshold ───────────────────────────────────────────────────
    if tau is not None:
        tau_used = float(tau)
        print(f"[postprocess] Using supplied τ = {tau_used:.4f}")
    elif label_col is not None and label_col in scored_df.columns:
        print("[postprocess] Running τ* sweep on labeled data...")
        tau_used = find_tau_star(
            scored_df,
            label_col=label_col,
            score_col=score_col,
            group_col=group_col,
        )
    else:
        tau_used = TAU_DEFAULT
        print(f"[postprocess] No GT labels and no τ supplied → using default τ = {tau_used:.4f}")

    # ── Apply threshold + singleton gate ─────────────────────────────────────
    print(f"\n[postprocess] Applying threshold τ = {tau_used:.4f} + singleton gate...")
    predictions = apply_threshold(
        scored_df,
        tau=tau_used,
        score_col=score_col,
        group_col=group_col,
        cand_id_col=cand_id_col,
    )

    n_matched   = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    print(f"[postprocess] Predictions: {n_matched} matched, {n_singleton} singleton")

    # ── Write candidate_pairs.tsv ─────────────────────────────────────────────
    candidate_path = output_dir / "candidate_pairs.tsv"
    write_candidate_pairs(
        scored_df,
        output_path=candidate_path,
        score_col=score_col,
        group_col=group_col,
        cand_id_col=cand_id_col,
    )

    # ── Write matching_results.tsv ────────────────────────────────────────────
    matching_path = output_dir / "matching_results.tsv"
    write_matching_results(
        predictions=predictions,
        all_s1_ids=all_s1_ids,
        output_path=matching_path,
    )

    return predictions, tau_used


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=" * 70)
    print("postprocess.py smoke test — task 4.2")
    print("=" * 70)

    # ── Build synthetic scored DataFrame ─────────────────────────────────────
    # Group A: 3 candidates, scores [0.9, 0.6, 0.3], labels [1, 0, 0]
    #   → at τ=0.78: max(score)=0.9 ≥ τ → predict candidates with score ≥ 0.78 → [A1]
    # Group B: 3 candidates, scores [0.4, 0.3, 0.2], labels [0, 0, 0]
    #   → at τ=0.78: max(score)=0.4 < τ → singleton gate → predict []
    # Group C: 2 candidates, scores [0.85, 0.75], labels [1, 1]
    #   → at τ=0.78: max(score)=0.85 ≥ τ → predict candidates with score ≥ 0.78 → [C1]
    #   (C2 has score=0.75 < 0.78 so it's not included at τ=0.78)

    scored_df = pl.DataFrame({
        "s1_id":  ["A", "A", "A", "B", "B", "B", "C", "C"],
        "cand_id":["A1","A2","A3","B1","B2","B3","C1","C2"],
        "score":  [0.9, 0.6, 0.3, 0.4, 0.3, 0.2, 0.85, 0.75],
        "label":  [1,   0,   0,   0,   0,   0,   1,    1],
    })

    all_s1_ids = ["A", "B", "C"]

    # ── 1. find_tau_star ──────────────────────────────────────────────────────
    print("\n--- Test 1: find_tau_star ---")
    tau_star = find_tau_star(scored_df, label_col="label", score_col="score")
    assert tau_star < 0.95, f"tau_star={tau_star} should be < 0.95"
    print(f"\n✓ tau_star={tau_star:.2f} < 0.95")

    # ── 2. apply_threshold at τ=0.78 ─────────────────────────────────────────
    print("\n--- Test 2: apply_threshold(tau=0.78) ---")
    preds = apply_threshold(scored_df, tau=0.78, score_col="score")

    # Group B must be singleton
    assert preds["B"] == [], f"Group B should predict [] (singleton gate), got {preds['B']}"
    print(f"✓ Group B = [] (singleton gate)")

    # Group A must include A1 (score=0.9 ≥ 0.78) but not A2 (0.6 < 0.78) or A3 (0.3)
    assert "A1" in preds["A"], f"A1 should be predicted for Group A, got {preds['A']}"
    assert "A2" not in preds["A"], f"A2 should NOT be predicted (0.6 < 0.78), got {preds['A']}"
    print(f"✓ Group A = {preds['A']} (contains A1, not A2/A3)")

    # Group C must include C1 (score=0.85 ≥ 0.78)
    assert "C1" in preds["C"], f"C1 should be predicted for Group C, got {preds['C']}"
    print(f"✓ Group C = {preds['C']}")

    # ── 3. sweep_threshold ───────────────────────────────────────────────────
    print("\n--- Test 3: sweep_threshold ---")
    tau_star2, curve = sweep_threshold(
        scored_df,
        label_col="label",
        score_col="score",
    )
    assert len(curve) == round((TAU_MAX - TAU_MIN) / TAU_STEP) + 1, \
        f"Expected {round((TAU_MAX - TAU_MIN) / TAU_STEP) + 1} curve points, got {len(curve)}"
    taus_in_curve = [t for t, _ in curve]
    assert abs(taus_in_curve[0] - TAU_MIN) < 1e-9, "First tau should be TAU_MIN"
    assert abs(taus_in_curve[-1] - TAU_MAX) < 1e-9, f"Last tau should be TAU_MAX, got {taus_in_curve[-1]}"
    print(f"✓ sweep_threshold: {len(curve)} points, tau* = {tau_star2:.2f}")

    # ── 4. Write outputs and verify format ────────────────────────────────────
    print("\n--- Test 4: write and verify output files ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)

        # Use postprocess() end-to-end
        result_preds, tau_used = postprocess(
            scored_df=scored_df,
            all_s1_ids=all_s1_ids,
            output_dir=tmp_path,
            tau=0.78,
            score_col="score",
            group_col="s1_id",
            cand_id_col="cand_id",
        )

        # ── Verify matching_results.tsv ───────────────────────────────────────
        matching_path = tmp_path / "matching_results.tsv"
        assert matching_path.exists(), "matching_results.tsv not written"

        with open(matching_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Check header
        header = lines[0].rstrip("\n")
        assert header == "source1_entity_id\tmatched_entity_ids", \
            f"Wrong matching header: {header!r}"
        print(f"✓ matching_results.tsv header correct: {header!r}")

        # Check all S1 ids present
        data_lines = lines[1:]
        found_s1s = {line.split("\t")[0] for line in data_lines if line.strip()}
        for sid in all_s1_ids:
            assert sid in found_s1s, f"S1 id {sid!r} missing from matching_results.tsv"
        print(f"✓ All {len(all_s1_ids)} S1 ids present in matching_results.tsv")

        # Check Group B is empty (singleton)
        b_line = next(l for l in data_lines if l.startswith("B\t"))
        b_val = b_line.rstrip("\n").split("\t")[1]
        assert b_val == "", f"Group B should be empty in matching_results, got {b_val!r}"
        print(f"✓ Group B row is empty (singleton) in matching_results.tsv")

        # Check Group A has A1
        a_line = next(l for l in data_lines if l.startswith("A\t"))
        a_val = a_line.rstrip("\n").split("\t")[1]
        assert "A1" in a_val.split(","), \
            f"Group A should contain A1 in matching_results, got {a_val!r}"
        print(f"✓ Group A contains A1 in matching_results.tsv: {a_val!r}")

        # ── Verify candidate_pairs.tsv ────────────────────────────────────────
        candidate_path = tmp_path / "candidate_pairs.tsv"
        assert candidate_path.exists(), "candidate_pairs.tsv not written"

        with open(candidate_path, encoding="utf-8") as f:
            cand_lines = f.readlines()

        # Check header (validator expects: source1_entity_id, candidate_entity_ids)
        cand_header = cand_lines[0].rstrip("\n")
        assert cand_header == "source1_entity_id\tcandidate_entity_ids", \
            f"Wrong candidate header: {cand_header!r}"
        print(f"✓ candidate_pairs.tsv header correct: {cand_header!r}")

        # Check all S1 ids have a candidate row
        cand_data = cand_lines[1:]
        found_cand_s1s = {line.split("\t")[0] for line in cand_data if line.strip()}
        for sid in all_s1_ids:
            assert sid in found_cand_s1s, \
                f"S1 id {sid!r} missing from candidate_pairs.tsv"
        print(f"✓ All {len(all_s1_ids)} S1 ids present in candidate_pairs.tsv")

        # Check Group A has all 3 candidates in candidate_pairs (no threshold applied)
        a_cand_line = next(l for l in cand_data if l.startswith("A\t"))
        a_cands = a_cand_line.rstrip("\n").split("\t")[1].split(",")
        assert set(a_cands) == {"A1", "A2", "A3"}, \
            f"Group A candidates should be {{A1,A2,A3}}, got {set(a_cands)}"
        print(f"✓ Group A has all 3 candidates in candidate_pairs.tsv: {a_cands}")

        print(f"\n✓ tau_used = {tau_used:.4f}")

    # ── 5. apply_threshold with no scored_df rows for a group ────────────────
    # (all_s1_ids can have S1 ids not present in scored_df — write_matching_results
    #  handles those as singletons via predictions.get(sid, []))
    print("\n--- Test 5: write_matching_results with extra S1 ids not in predictions ---")
    with tempfile.TemporaryDirectory() as tmpdir2:
        tmp2 = pathlib.Path(tmpdir2)
        extra_ids = ["A", "B", "C", "D", "E"]  # D and E have no predictions
        write_matching_results(
            predictions={"A": ["A1"], "B": [], "C": ["C1"]},
            all_s1_ids=extra_ids,
            output_path=tmp2 / "matching_results.tsv",
        )
        with open(tmp2 / "matching_results.tsv", encoding="utf-8") as f:
            lines2 = f.readlines()
        data2 = {l.split("\t")[0]: l.rstrip("\n").split("\t")[1]
                 for l in lines2[1:] if l.strip()}
        assert "D" in data2, "D should appear in output even without prediction"
        assert data2["D"] == "", f"D should be empty string, got {data2['D']!r}"
        assert "E" in data2, "E should appear in output even without prediction"
        assert data2["E"] == "", f"E should be empty string, got {data2['E']!r}"
        print(f"✓ Extra S1 ids D, E written as empty singletons")

    print()
    print("=" * 70)
    print("postprocess.py smoke test PASSED ✓")
    print("=" * 70)
