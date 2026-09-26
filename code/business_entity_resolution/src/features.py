"""
features.py — Feature engineering for Business Entity Resolution.

Responsibilities (task 3.1):
  - String similarity features (9 columns):
      lev_norm_name, lev_norm_addr    — rapidfuzz Levenshtein normalized
      jw_name, jw_addr                — rapidfuzz Jaro-Winkler
      token_sort_name                 — rapidfuzz token_sort_ratio
      monge_elkan_name                — textdistance Monge-Elkan (handles transposition)
      jaccard_3gram_name              — char 3-gram Jaccard
      token_jaccard_name              — token-set Jaccard
      containment_name                — asymmetric containment C(T1,T2)=|∩|/|T1|

Responsibilities (task 3.2 — full 28-dim vector):
  - TF-IDF cosine sublinear 1-2gram (name + addr)
  - I_country (0/1) — open-set, pure string equality
  - Numeric Jaccard + I_num_conflict (Shop 12 vs 13)
  - Dense cosine v1^T vx — paraphrase-multilingual-MiniLM-L12-v2
  - Address string metrics: token_sort, monge_elkan, jaccard_3gram, token_jaccard,
    containment, addr_containment_rev
  - Cross-field: lev_norm_name_addr, jw_name_addr, containment_name2addr
  - Length features: len_diff_name, len_diff_addr, token_count_diff_name, token_count_diff_addr

Design note on vectorization:
  Polars map_elements is C-backed and runs Python UDFs per element.
  For 50k-row chunks this is fast enough (<2s per chunk).
  TF-IDF and dense features use batch numpy/sklearn paths over the whole chunk.
"""

from __future__ import annotations

import re
from typing import Callable

import polars as pl
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz.distance import JaroWinkler as rf_JW
from rapidfuzz.distance import Levenshtein as rf_Lev

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe(s: str | None) -> str:
    """Coerce None/null to empty string."""
    if s is None:
        return ""
    return str(s)


def _char_3grams(text: str) -> frozenset[str]:
    """All character 3-grams of a string."""
    if len(text) < 3:
        return frozenset()
    return frozenset(text[i:i+3] for i in range(len(text) - 2))


# ── Per-pair scalar metrics ───────────────────────────────────────────────────

def lev_norm(s1: str, s2: str) -> float:
    """Normalized Levenshtein distance.

    Returns 0.0 for identical strings, 1.0 for completely different.
    = 1 - rapidfuzz Levenshtein normalized_similarity
    """
    s1, s2 = _safe(s1), _safe(s2)
    if not s1 and not s2:
        return 0.0
    return 1.0 - rf_Lev.normalized_similarity(s1, s2)


def jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity in [0, 1]. 1.0 = identical.

    Uses rapidfuzz with default prefix weight p=0.1.
    """
    s1, s2 = _safe(s1), _safe(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return rf_JW.normalized_similarity(s1, s2)


def token_sort(s1: str, s2: str) -> float:
    """Token sort ratio in [0, 1].

    Sorts tokens alphabetically then computes similarity — handles word-order
    variations like "Tata Motors" vs "Motors Tata".
    """
    s1, s2 = _safe(s1), _safe(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return rf_fuzz.token_sort_ratio(s1, s2) / 100.0


def monge_elkan(s1: str, s2: str) -> float:
    """Monge-Elkan similarity.

    ME(A, B) = 1/|A| * Σ_i max_j JW(a_i, b_j)
    where A = s1.split(), B = s2.split().

    Handles token transposition better than token_sort for partial overlaps
    (e.g., "Tata Motors" vs "Motors Tata Ltd").
    Returns 0.0 if either string is empty.
    """
    s1, s2 = _safe(s1), _safe(s2)
    if not s1 or not s2:
        return 0.0
    tokens_a = s1.split()
    tokens_b = s2.split()
    if not tokens_a or not tokens_b:
        return 0.0
    total = 0.0
    for a in tokens_a:
        best = max(rf_JW.normalized_similarity(a, b) for b in tokens_b)
        total += best
    return total / len(tokens_a)


def jaccard_3gram(s1: str, s2: str) -> float:
    """Char 3-gram Jaccard similarity.

    J3 = |gram3(s1) ∩ gram3(s2)| / |gram3(s1) ∪ gram3(s2)|
    For strings shorter than 3 chars, falls back to exact match (1.0/0.0).
    """
    s1, s2 = _safe(s1), _safe(s2)
    g1 = _char_3grams(s1)
    g2 = _char_3grams(s2)
    # Short-string fallback
    if not g1 and not g2:
        return 1.0 if s1 == s2 else 0.0
    if not g1 or not g2:
        return 0.0
    inter = len(g1 & g2)
    union = len(g1 | g2)
    return inter / union if union > 0 else 0.0


def token_jaccard(s1: str, s2: str) -> float:
    """Token-set Jaccard similarity.

    J = |tok1 ∩ tok2| / |tok1 ∪ tok2|
    where tok1 = set(s1.split()), tok2 = set(s2.split())
    """
    s1, s2 = _safe(s1), _safe(s2)
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union > 0 else 0.0


def containment(s1: str, s2: str) -> float:
    """Asymmetric token containment C(T1, T2) = |tok1 ∩ tok2| / |tok1|.

    Measures how much of T1 is covered by T2 — good for abbreviated addresses
    where T1 is the query (shorter) and T2 is the candidate.
    Returns 0.0 if tok1 is empty.
    """
    s1, s2 = _safe(s1), _safe(s2)
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    if not t1:
        return 0.0
    return len(t1 & t2) / len(t1)


# ── Vectorized batch computation via Polars map_elements ─────────────────────

def _make_series_applier(fn: Callable[[str, str], float]):
    """Return a function that applies fn(a, b) elementwise over two Series."""
    def apply(col_a: pl.Series, col_b: pl.Series) -> pl.Series:
        # zip both series into structs, map over them
        return pl.Series(
            name="_tmp",
            values=[
                fn(_safe(a), _safe(b))
                for a, b in zip(col_a.to_list(), col_b.to_list())
            ],
            dtype=pl.Float32,
        )
    return apply


_apply_lev = _make_series_applier(lev_norm)
_apply_jw = _make_series_applier(jaro_winkler)
_apply_ts = _make_series_applier(token_sort)
_apply_me = _make_series_applier(monge_elkan)
_apply_j3 = _make_series_applier(jaccard_3gram)
_apply_tj = _make_series_applier(token_jaccard)
_apply_ct = _make_series_applier(containment)


def compute_string_features(
    df: pl.DataFrame,
    name1_col: str = "name1",
    name2_col: str = "name2",
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
) -> pl.DataFrame:
    """Compute all 9 string similarity features and append as new columns.

    Parameters
    ----------
    df : pl.DataFrame
        Input chunk (typically 50k rows) with name/address pair columns.
    name1_col, name2_col : str
        Column names for the S1 and S2/S3 business names (preprocessed).
    addr1_col, addr2_col : str
        Column names for the S1 and S2/S3 addresses (preprocessed).

    Returns
    -------
    pl.DataFrame
        Original columns plus 9 new Float32 columns:
          lev_norm_name, lev_norm_addr,
          jw_name, jw_addr,
          token_sort_name,
          monge_elkan_name,
          jaccard_3gram_name,
          token_jaccard_name,
          containment_name
    """
    n1 = df[name1_col]
    n2 = df[name2_col]
    a1 = df[addr1_col]
    a2 = df[addr2_col]

    lev_name_vals   = _apply_lev(n1, n2)
    lev_addr_vals   = _apply_lev(a1, a2)
    jw_name_vals    = _apply_jw(n1, n2)
    jw_addr_vals    = _apply_jw(a1, a2)
    ts_name_vals    = _apply_ts(n1, n2)
    me_name_vals    = _apply_me(n1, n2)
    j3_name_vals    = _apply_j3(n1, n2)
    tj_name_vals    = _apply_tj(n1, n2)
    ct_name_vals    = _apply_ct(n1, n2)

    return df.with_columns([
        lev_name_vals.alias("lev_norm_name"),
        lev_addr_vals.alias("lev_norm_addr"),
        jw_name_vals.alias("jw_name"),
        jw_addr_vals.alias("jw_addr"),
        ts_name_vals.alias("token_sort_name"),
        me_name_vals.alias("monge_elkan_name"),
        j3_name_vals.alias("jaccard_3gram_name"),
        tj_name_vals.alias("token_jaccard_name"),
        ct_name_vals.alias("containment_name"),
    ])


# ── String feature column names (used by model.py and pipeline.py) ───────────
STRING_FEATURE_COLS = [
    "lev_norm_name",
    "lev_norm_addr",
    "jw_name",
    "jw_addr",
    "token_sort_name",
    "monge_elkan_name",
    "jaccard_3gram_name",
    "token_jaccard_name",
    "containment_name",
]


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("features.py smoke test — task 3.1")
    print("=" * 70)

    cases = [
        ("Identical", "abc bank limited", "abc bank limited",
         "abc bank 123 main st", "abc bank 123 main st"),
        ("Completely different", "xyz corp", "delta industries",
         "100 oak avenue", "500 pine street"),
        ("Transposition", "tata motors", "motors tata",
         "pune industrial area", "industrial area pune"),
        ("Abbreviated address", "sbi atm branch", "sbi",
         "sbi atm branch main road", "sbi"),
        ("French diacritic", "societe generale", "societe generale sarl",
         "12 rue de la paix paris 75001", "12 rue de la paix paris"),
        ("India transliteration", "choudhury and sons", "chowdhury sons",
         "near sbi atm mumbai 400001", "near sbi atm mumbai"),
        ("Empty vs non-empty", "", "abc corp",
         "", "123 main st"),
    ]

    # Column headers
    col_w = 22
    metrics = [
        "lev_norm", "jaro_winkler", "token_sort",
        "monge_elkan", "jaccard_3gram", "token_jaccard", "containment",
    ]
    header = f"{'Case':<24}" + "".join(f"{m:>14}" for m in metrics)
    print(f"\n{header}")
    print("-" * (24 + 14 * len(metrics)))

    for label, n1, n2, a1, a2 in cases:
        lv  = lev_norm(n1, n2)
        jw  = jaro_winkler(n1, n2)
        ts  = token_sort(n1, n2)
        me  = monge_elkan(n1, n2)
        j3  = jaccard_3gram(n1, n2)
        tj  = token_jaccard(n1, n2)
        ct  = containment(n1, n2)
        row = f"{label:<24}" + "".join(f"{v:>14.4f}" for v in [lv, jw, ts, me, j3, tj, ct])
        print(row)

    print()

    # ── Assertions ────────────────────────────────────────────────────────

    # 1. Identical → all similarity metrics ≈ 1.0 (lev_norm ≈ 0.0 since it's distance)
    n = "abc bank limited"
    assert lev_norm(n, n) == 0.0, "lev_norm identical should be 0.0"
    assert jaro_winkler(n, n) == 1.0, "jaro_winkler identical should be 1.0"
    assert token_sort(n, n) == 1.0, "token_sort identical should be 1.0"
    assert monge_elkan(n, n) == 1.0, "monge_elkan identical should be 1.0"
    assert jaccard_3gram(n, n) == 1.0, "jaccard_3gram identical should be 1.0"
    assert token_jaccard(n, n) == 1.0, "token_jaccard identical should be 1.0"
    assert containment(n, n) == 1.0, "containment identical should be 1.0"
    print("✓ Identical strings: all metrics correct")

    # 2. Transposition: Monge-Elkan should be higher than token_sort would imply poor matching
    s1, s2 = "tata motors", "motors tata"
    me_val = monge_elkan(s1, s2)
    ts_val = token_sort(s1, s2)
    # Both should be high since both handle transposition, but ME is asymmetric
    assert me_val > 0.8, f"monge_elkan transposition should be >0.8, got {me_val:.4f}"
    assert ts_val > 0.8, f"token_sort transposition should be >0.8, got {ts_val:.4f}"
    print(f"✓ Transposition ('tata motors' vs 'motors tata'): ME={me_val:.4f}, TS={ts_val:.4f}")

    # 3. Containment > token_jaccard for "SBI ATM Branch" vs "SBI"
    s1, s2 = "sbi atm branch", "sbi"
    ct_val = containment(s2, s1)  # C(T2="sbi", T1="sbi atm branch") — how much of s2 is in s1
    tj_val = token_jaccard(s1, s2)
    assert ct_val > tj_val, (
        f"containment({s2!r} vs {s1!r})={ct_val:.4f} should > token_jaccard={tj_val:.4f}"
    )
    print(f"✓ Containment > token_jaccard: CT={ct_val:.4f}, TJ={tj_val:.4f}")

    # 4. Empty string edge cases
    assert lev_norm("", "") == 0.0
    assert jaro_winkler("", "") == 1.0
    assert token_sort("", "") == 1.0
    assert monge_elkan("", "abc") == 0.0
    assert jaccard_3gram("", "") == 1.0  # both empty → exact match
    assert token_jaccard("", "") == 1.0
    assert containment("", "abc") == 0.0
    print("✓ Empty string edge cases handled")

    # 5. Short string (<3 chars) 3-gram fallback
    assert jaccard_3gram("ab", "ab") == 1.0, "short identical → 1.0"
    assert jaccard_3gram("ab", "cd") == 0.0, "short different → 0.0"
    print("✓ Short string 3-gram fallback works")

    # 6. compute_string_features Polars batch
    import polars as pl

    test_df = pl.DataFrame({
        "name1": ["abc bank", "tata motors", "sbi atm branch", ""],
        "name2": ["abc bank", "motors tata", "sbi", "xyz"],
        "addr1": ["123 main st", "pune area", "main road", ""],
        "addr2": ["123 main st", "area pune", "main road branch", "oak ave"],
    })

    result_df = compute_string_features(test_df)

    # Check all 9 columns present
    for col in STRING_FEATURE_COLS:
        assert col in result_df.columns, f"Missing column: {col}"
    assert len(result_df.columns) == len(test_df.columns) + len(STRING_FEATURE_COLS)

    print(f"\n✓ compute_string_features output ({len(result_df)} rows × {len(result_df.columns)} cols):")
    print(result_df.select(STRING_FEATURE_COLS))

    # Check identical row (row 0) → lev_norm ≈ 0, rest ≈ 1
    row0 = result_df.row(0, named=True)
    assert row0["lev_norm_name"] < 0.01, f"identical lev_norm_name should be ~0: {row0['lev_norm_name']}"
    assert row0["jw_name"] > 0.99, f"identical jw_name should be ~1: {row0['jw_name']}"
    print("\n✓ Polars batch function verified — identical pair row correct")

    # 7. 50k chunk streaming test
    import random
    random.seed(42)
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa"]
    big_df = pl.DataFrame({
        "name1": [" ".join(random.choices(words, k=3)) for _ in range(50_000)],
        "name2": [" ".join(random.choices(words, k=3)) for _ in range(50_000)],
        "addr1": [" ".join(random.choices(words, k=4)) for _ in range(50_000)],
        "addr2": [" ".join(random.choices(words, k=4)) for _ in range(50_000)],
    })
    import time
    t0 = time.time()
    big_result = compute_string_features(big_df)
    elapsed = time.time() - t0
    assert len(big_result) == 50_000
    assert all(c in big_result.columns for c in STRING_FEATURE_COLS)
    print(f"\n✓ 50k chunk processed in {elapsed:.2f}s — {len(STRING_FEATURE_COLS)} feature cols generated")

    print("\n" + "=" * 70)
    print("All smoke tests PASSED ✓")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3.2 — Additional features to reach 28-dim total
# ═══════════════════════════════════════════════════════════════════════════════

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.preprocessing import normalize as sk_normalize

# ── Model lazy loading ────────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def _get_model():
    """Load paraphrase-multilingual-MiniLM-L12-v2 once and cache in module."""
    if "model" not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE["model"] = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _MODEL_CACHE["model"]


# ── Group A: TF-IDF cosine batch helper ──────────────────────────────────────

def _tfidf_cosine_batch(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Fit TF-IDF on all texts in chunk, return pairwise diagonal cosine sims.

    Parameters
    ----------
    texts_a, texts_b : equal-length lists of strings
        text_a[i] paired with text_b[i].

    Returns
    -------
    np.ndarray, shape (N,), dtype float32
        Cosine similarity for each pair.
    """
    n = len(texts_a)
    if n == 0:
        return np.array([], dtype=np.float32)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=50_000,
    )
    all_texts = texts_a + texts_b
    tfidf_matrix = vectorizer.fit_transform(all_texts)  # (2N, vocab)
    mat_a = tfidf_matrix[:n]
    mat_b = tfidf_matrix[n:]
    # Row-normalize → element-wise product → row sum = diagonal of cosine sim
    mat_a_norm = sk_normalize(mat_a, norm="l2")
    mat_b_norm = sk_normalize(mat_b, norm="l2")
    scores = np.array(mat_a_norm.multiply(mat_b_norm).sum(axis=1)).flatten()
    return scores.astype(np.float32)


def compute_tfidf_features(
    df: pl.DataFrame,
    name1_col: str = "name1",
    name2_col: str = "name2",
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
) -> pl.DataFrame:
    """Add tfidf_cosine_name and tfidf_cosine_addr columns.

    Fits one TF-IDF vectorizer per field per chunk over all texts in the chunk,
    then computes pairwise diagonal cosine similarities (no per-row loop).
    """
    names_a = [_safe(v) for v in df[name1_col].to_list()]
    names_b = [_safe(v) for v in df[name2_col].to_list()]
    addrs_a = [_safe(v) for v in df[addr1_col].to_list()]
    addrs_b = [_safe(v) for v in df[addr2_col].to_list()]

    tfidf_name = _tfidf_cosine_batch(names_a, names_b)
    tfidf_addr = _tfidf_cosine_batch(addrs_a, addrs_b)

    return df.with_columns([
        pl.Series("tfidf_cosine_name", tfidf_name, dtype=pl.Float32),
        pl.Series("tfidf_cosine_addr", tfidf_addr, dtype=pl.Float32),
    ])


# ── Group B: Country indicator ────────────────────────────────────────────────

def compute_country_features(
    df: pl.DataFrame,
    country1_col: str = "country1",
    country2_col: str = "country2",
) -> pl.DataFrame:
    """Add i_country column.

    1.0 if lower(country1) == lower(country2), else 0.0.
    Null/None treated as empty string. Open-set: no hardcoded list.
    """
    c1 = [_safe(v).lower() for v in df[country1_col].to_list()]
    c2 = [_safe(v).lower() for v in df[country2_col].to_list()]
    vals = np.array(
        [1.0 if a == b else 0.0 for a, b in zip(c1, c2)],
        dtype=np.float32,
    )
    return df.with_columns([
        pl.Series("i_country", vals, dtype=pl.Float32),
    ])


# ── Group C: Numeric features ─────────────────────────────────────────────────

_NUM_RE = re.compile(r"\d+")


def _extract_nums(text: str) -> frozenset[str]:
    """Extract all digit sequences from text as a frozenset of strings."""
    return frozenset(_NUM_RE.findall(text))


def _numeric_pair(n1: str, a1: str, n2: str, a2: str) -> tuple[float, float]:
    """Return (numeric_jaccard, i_num_conflict) for one pair.

    N1 = digit sequences from name1 + addr1
    Nx = digit sequences from name2 + addr2
    numeric_jaccard = |N1 ∩ Nx| / |N1 ∪ Nx|, 0.0 if both empty
    i_num_conflict = 1.0 if N1≠∅ AND Nx≠∅ AND N1 ∩ Nx = ∅
    """
    N1 = _extract_nums(f"{n1} {a1}")
    Nx = _extract_nums(f"{n2} {a2}")
    union = N1 | Nx
    inter = N1 & Nx

    if not union:
        nj = 0.0
    else:
        nj = len(inter) / len(union)

    if N1 and Nx and not inter:
        inc = 1.0
    else:
        inc = 0.0

    return nj, inc


def compute_numeric_features(
    df: pl.DataFrame,
    name1_col: str = "name1",
    name2_col: str = "name2",
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
) -> pl.DataFrame:
    """Add numeric_jaccard and i_num_conflict columns."""
    n1s = df[name1_col].to_list()
    a1s = df[addr1_col].to_list()
    n2s = df[name2_col].to_list()
    a2s = df[addr2_col].to_list()

    nj_vals = np.empty(len(df), dtype=np.float32)
    inc_vals = np.empty(len(df), dtype=np.float32)

    for i, (n1, a1, n2, a2) in enumerate(zip(n1s, a1s, n2s, a2s)):
        nj, inc = _numeric_pair(
            _safe(n1), _safe(a1), _safe(n2), _safe(a2)
        )
        nj_vals[i] = nj
        inc_vals[i] = inc

    return df.with_columns([
        pl.Series("numeric_jaccard", nj_vals, dtype=pl.Float32),
        pl.Series("i_num_conflict", inc_vals, dtype=pl.Float32),
    ])


# ── Group D: Dense cosine ─────────────────────────────────────────────────────

def _dense_cosine_batch(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Encode texts, L2-normalize, return dot product per pair.

    De-duplicates texts before encoding to avoid re-encoding repeated values.
    Uses batch_size=256 to stay within memory limits.

    Returns np.ndarray shape (N,) dtype float32.
    """
    n = len(texts_a)
    if n == 0:
        return np.array([], dtype=np.float32)

    model = _get_model()
    # Collect unique texts across both sides
    all_unique = list(dict.fromkeys(texts_a + texts_b))  # preserve insertion order, dedup
    embeddings = model.encode(
        all_unique,
        batch_size=256,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    emb_map = {t: emb for t, emb in zip(all_unique, embeddings)}

    vecs_a = np.vstack([emb_map[t] for t in texts_a])
    vecs_b = np.vstack([emb_map[t] for t in texts_b])
    vecs_a = sk_normalize(vecs_a, norm="l2")
    vecs_b = sk_normalize(vecs_b, norm="l2")
    scores = (vecs_a * vecs_b).sum(axis=1)
    return scores.astype(np.float32)


def compute_dense_features(
    df: pl.DataFrame,
    name1_col: str = "name1",
    name2_col: str = "name2",
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
) -> pl.DataFrame:
    """Add dense_cosine column using paraphrase-multilingual-MiniLM-L12-v2.

    Encodes name1 and name2 (not addresses — name is the primary identity field).
    """
    names_a = [_safe(v) for v in df[name1_col].to_list()]
    names_b = [_safe(v) for v in df[name2_col].to_list()]
    scores = _dense_cosine_batch(names_a, names_b)
    return df.with_columns([
        pl.Series("dense_cosine", scores, dtype=pl.Float32),
    ])


# ── Group E: Address string metrics (6 cols) ──────────────────────────────────

def compute_address_string_features(
    df: pl.DataFrame,
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
) -> pl.DataFrame:
    """Add 6 address-pair string metrics.

    Columns added:
        token_sort_addr      — token_sort(addr1, addr2)
        monge_elkan_addr     — monge_elkan(addr1, addr2)
        jaccard_3gram_addr   — jaccard_3gram(addr1, addr2)
        token_jaccard_addr   — token_jaccard(addr1, addr2)
        containment_addr     — containment(addr1, addr2)  C(addr1,addr2)
        addr_containment_rev — containment(addr2, addr1)  C(addr2,addr1)
    """
    a1 = df[addr1_col]
    a2 = df[addr2_col]

    return df.with_columns([
        _make_series_applier(token_sort)(a1, a2).alias("token_sort_addr"),
        _make_series_applier(monge_elkan)(a1, a2).alias("monge_elkan_addr"),
        _make_series_applier(jaccard_3gram)(a1, a2).alias("jaccard_3gram_addr"),
        _make_series_applier(token_jaccard)(a1, a2).alias("token_jaccard_addr"),
        _make_series_applier(containment)(a1, a2).alias("containment_addr"),
        _make_series_applier(containment)(a2, a1).alias("addr_containment_rev"),
    ])


# ── Group F: Cross-field features (3 cols) ────────────────────────────────────

def compute_cross_field_features(
    df: pl.DataFrame,
    name1_col: str = "name1",
    name2_col: str = "name2",
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
) -> pl.DataFrame:
    """Add 3 cross-field features.

    Columns added:
        lev_norm_name_addr    — lev_norm(name1+' '+addr1, name2+' '+addr2)
        jw_name_addr          — jaro_winkler(name1+' '+addr1, name2+' '+addr2)
        containment_name2addr — containment(name1, addr2) — is name1 in addr2?
    """
    n1 = df[name1_col].to_list()
    n2 = df[name2_col].to_list()
    a1 = df[addr1_col].to_list()
    a2 = df[addr2_col].to_list()

    combined1 = [f"{_safe(n)} {_safe(a)}" for n, a in zip(n1, a1)]
    combined2 = [f"{_safe(n)} {_safe(a)}" for n, a in zip(n2, a2)]

    c1_series = pl.Series("_c1", combined1)
    c2_series = pl.Series("_c2", combined2)
    n1_series = df[name1_col]
    a2_series = df[addr2_col]

    lev_na  = _make_series_applier(lev_norm)(c1_series, c2_series)
    jw_na   = _make_series_applier(jaro_winkler)(c1_series, c2_series)
    ct_n2a  = _make_series_applier(containment)(n1_series, a2_series)

    return df.with_columns([
        lev_na.alias("lev_norm_name_addr"),
        jw_na.alias("jw_name_addr"),
        ct_n2a.alias("containment_name2addr"),
    ])


# ── Group G: Length features (4 cols) ────────────────────────────────────────

def _len_diff(s1: str, s2: str) -> float:
    """Normalized length difference: abs(len(s1)-len(s2)) / max(len(s1),len(s2),1)."""
    l1, l2 = len(s1), len(s2)
    return abs(l1 - l2) / max(l1, l2, 1)


def _token_count_diff(s1: str, s2: str) -> float:
    """Normalized token count difference."""
    t1, t2 = len(s1.split()), len(s2.split())
    return abs(t1 - t2) / max(t1, t2, 1)


def compute_length_features(
    df: pl.DataFrame,
    name1_col: str = "name1",
    name2_col: str = "name2",
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
) -> pl.DataFrame:
    """Add len_diff_name, len_diff_addr, token_count_diff_name, token_count_diff_addr."""
    n1 = df[name1_col]
    n2 = df[name2_col]
    a1 = df[addr1_col]
    a2 = df[addr2_col]

    return df.with_columns([
        _make_series_applier(_len_diff)(n1, n2).alias("len_diff_name"),
        _make_series_applier(_len_diff)(a1, a2).alias("len_diff_addr"),
        _make_series_applier(_token_count_diff)(n1, n2).alias("token_count_diff_name"),
        _make_series_applier(_token_count_diff)(a1, a2).alias("token_count_diff_addr"),
    ])


# ── Authoritative 28-feature column list ─────────────────────────────────────

ALL_FEATURE_COLS: list[str] = [
    # --- Group 3.1: string metrics (9) ---
    "lev_norm_name",        # 0
    "lev_norm_addr",        # 1
    "jw_name",              # 2
    "jw_addr",              # 3
    "token_sort_name",      # 4
    "monge_elkan_name",     # 5
    "jaccard_3gram_name",   # 6
    "token_jaccard_name",   # 7
    "containment_name",     # 8
    # --- Group 3.2a: TF-IDF cosine (2) ---
    "tfidf_cosine_name",    # 9
    "tfidf_cosine_addr",    # 10
    # --- Group 3.2b: country (1) ---
    "i_country",            # 11
    # --- Group 3.2c: numeric (2) ---
    "numeric_jaccard",      # 12
    "i_num_conflict",       # 13
    # --- Group 3.2d: dense embedding (1) ---
    "dense_cosine",         # 14
    # --- Group 3.2e: address string metrics (6) ---
    "token_sort_addr",      # 15
    "monge_elkan_addr",     # 16
    "jaccard_3gram_addr",   # 17
    "token_jaccard_addr",   # 18
    "containment_addr",     # 19
    "addr_containment_rev", # 20
    # --- Group 3.2f: cross-field (3) ---
    "lev_norm_name_addr",   # 21
    "jw_name_addr",         # 22
    "containment_name2addr",# 23
    # --- Group 3.2g: length features (4) ---
    "len_diff_name",        # 24
    "len_diff_addr",        # 25
    "token_count_diff_name",# 26
    "token_count_diff_addr",# 27
]

assert len(ALL_FEATURE_COLS) == 28, f"Expected 28 features, got {len(ALL_FEATURE_COLS)}"


# ── Master orchestrator ───────────────────────────────────────────────────────

def compute_all_features(
    df: pl.DataFrame,
    name1_col: str = "name1",
    name2_col: str = "name2",
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
    country1_col: str = "country1",
    country2_col: str = "country2",
) -> pl.DataFrame:
    """Master function: apply all feature groups in order → 28 new columns.

    Input df must have columns: name1, name2, addr1, addr2, country1, country2.
    Output df has all input columns PLUS the 28 feature columns in ALL_FEATURE_COLS order.

    Parameters
    ----------
    df : pl.DataFrame
        Candidate pairs chunk (typically ≤50k rows).
    name1_col, name2_col : str
        Column names for source1 and source2/3 business names (preprocessed).
    addr1_col, addr2_col : str
        Column names for source1 and source2/3 addresses (preprocessed).
    country1_col, country2_col : str
        Column names for country strings (raw, any case).

    Returns
    -------
    pl.DataFrame
        Original columns + 28 Float32 feature columns.
    """
    df = compute_string_features(df, name1_col, name2_col, addr1_col, addr2_col)
    df = compute_tfidf_features(df, name1_col, name2_col, addr1_col, addr2_col)
    df = compute_country_features(df, country1_col, country2_col)
    df = compute_numeric_features(df, name1_col, name2_col, addr1_col, addr2_col)
    df = compute_dense_features(df, name1_col, name2_col, addr1_col, addr2_col)
    df = compute_address_string_features(df, addr1_col, addr2_col)
    df = compute_cross_field_features(df, name1_col, name2_col, addr1_col, addr2_col)
    df = compute_length_features(df, name1_col, name2_col, addr1_col, addr2_col)
    return df


# ── Chunked streaming wrapper ─────────────────────────────────────────────────

def compute_features_chunked(
    pairs_df: pl.DataFrame,
    chunk_size: int = 50_000,
    name1_col: str = "name1",
    name2_col: str = "name2",
    addr1_col: str = "addr1",
    addr2_col: str = "addr2",
    country1_col: str = "country1",
    country2_col: str = "country2",
) -> pl.DataFrame:
    """Process arbitrarily large pairs_df in chunks of chunk_size rows.

    Calls compute_all_features on each chunk, concatenates results, and
    returns a single DataFrame with all 28 feature columns appended.

    Uses tqdm progress bar when more than one chunk is needed.

    Parameters
    ----------
    pairs_df : pl.DataFrame
        Full candidate pairs table.
    chunk_size : int
        Rows per chunk; default 50_000 to stay ≤16GB RAM.

    Returns
    -------
    pl.DataFrame
        Full table with 28 feature columns appended.
    """
    from tqdm import tqdm

    n_rows = len(pairs_df)
    if n_rows == 0:
        # Return schema-correct empty frame
        return compute_all_features(
            pairs_df,
            name1_col, name2_col, addr1_col, addr2_col,
            country1_col, country2_col,
        )

    n_chunks = (n_rows + chunk_size - 1) // chunk_size
    chunks_out: list[pl.DataFrame] = []

    iterator = range(n_chunks)
    if n_chunks > 1:
        iterator = tqdm(iterator, desc="Feature chunks", unit="chunk")

    for i in iterator:
        start = i * chunk_size
        end = min(start + chunk_size, n_rows)
        chunk = pairs_df.slice(start, end - start)
        chunk_out = compute_all_features(
            chunk,
            name1_col, name2_col, addr1_col, addr2_col,
            country1_col, country2_col,
        )
        chunks_out.append(chunk_out)

    return pl.concat(chunks_out, rechunk=False)


# ── test_features() — Phase 3 gate function ──────────────────────────────────

def test_features() -> None:
    """Regression test for Phase 3 gate.

    Verifies:
    1. I_num_conflict fires correctly  (Shop 12 vs Shop 13 → 1.0)
    2. I_num_conflict stays 0.0 for same numbers
    3. I_num_conflict stays 0.0 when one side has no numbers
    4. compute_all_features returns exactly 28 feature columns
    5. 50k chunk streaming produces correct row count
    6. i_country works for France (open-set, unseen country)
    7. tfidf_cosine_name is in [0, 1]
    8. dense_cosine is NOT loaded — patched to zeros (fast, no GPU required)

    Raises AssertionError on failure. Prints PASS on success.
    """
    # ── Patch dense model so we don't load 450MB weights during testing ──────
    # We patch via globals() so the bare-name lookup inside compute_dense_features
    # picks up the mock.  Works whether the module is imported normally or loaded
    # directly via importlib / exec.

    _g = globals()

    def _mock_dense(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
        return np.zeros(len(texts_a), dtype=np.float32)

    _orig_fn = _g["_dense_cosine_batch"]
    _g["_dense_cosine_batch"] = _mock_dense

    try:
        # ── 1-3: I_num_conflict scalar tests ─────────────────────────────────
        nj, inc = _numeric_pair("shop 12", "main st", "shop 13", "main st")
        assert inc == 1.0, f"Shop 12 vs Shop 13: expected i_num_conflict=1.0, got {inc}"
        assert nj == 0.0, f"Shop 12 vs Shop 13: expected numeric_jaccard=0.0, got {nj}"
        print("✓ 1. I_num_conflict=1.0 for Shop 12 vs Shop 13")

        nj2, inc2 = _numeric_pair("shop 12", "main st", "shop 12", "main st")
        assert inc2 == 0.0, f"Same numbers: expected i_num_conflict=0.0, got {inc2}"
        assert nj2 == 1.0, f"Same numbers: expected numeric_jaccard=1.0, got {nj2}"
        print("✓ 2. I_num_conflict=0.0 for same numbers (Shop 12 vs Shop 12)")

        nj3, inc3 = _numeric_pair("abc corp", "", "abc corp", "")
        assert inc3 == 0.0, f"No numbers: expected i_num_conflict=0.0, got {inc3}"
        assert nj3 == 0.0, f"No numbers: expected numeric_jaccard=0.0, got {nj3}"
        print("✓ 3. I_num_conflict=0.0 when neither side has numbers")

        nj4, inc4 = _numeric_pair("shop 12", "", "abc corp", "")
        assert inc4 == 0.0, f"One side no numbers: expected i_num_conflict=0.0, got {inc4}"
        print("✓ 3b. I_num_conflict=0.0 when only one side has numbers")

        # ── 4: compute_all_features returns exactly 28 feature columns ────────
        test_df = pl.DataFrame({
            "name1":    ["shop 12 main st", "société générale", "tata motors", "abc corp"],
            "name2":    ["shop 13 main st", "societe generale sarl", "motors tata", "abc corp"],
            "addr1":    ["12 main street", "12 rue de la paix paris", "pune area", ""],
            "addr2":    ["13 main street", "12 rue de la paix", "area pune", "oak avenue"],
            "country1": ["us", "france", "india", "us"],
            "country2": ["us", "france", "india", "india"],
        })
        result = compute_all_features(test_df)
        new_cols = [c for c in result.columns if c in ALL_FEATURE_COLS]
        assert len(new_cols) == 28, f"Expected 28 feature cols, got {len(new_cols)}: {new_cols}"
        print(f"✓ 4. compute_all_features adds exactly 28 feature columns")

        # Verify each expected column is present
        for col in ALL_FEATURE_COLS:
            assert col in result.columns, f"Missing feature column: {col}"
        print("✓ 4b. All 28 ALL_FEATURE_COLS present in output")

        # ── 5: i_country — open-set France ───────────────────────────────────
        france_row = result.filter(pl.col("country1") == "france")
        assert len(france_row) == 1
        ic = france_row["i_country"][0]
        assert ic == 1.0, f"France vs France: expected i_country=1.0, got {ic}"
        print("✓ 5. i_country=1.0 for France vs France (open-set, unseen country)")

        us_india_row = result.filter(
            (pl.col("country1") == "us") & (pl.col("country2") == "india")
        )
        assert len(us_india_row) == 1
        ic2 = us_india_row["i_country"][0]
        assert ic2 == 0.0, f"US vs India: expected i_country=0.0, got {ic2}"
        print("✓ 5b. i_country=0.0 for US vs India")

        # ── 6: tfidf_cosine_name in [0, 1] ───────────────────────────────────
        tfidf_vals = result["tfidf_cosine_name"].to_numpy()
        assert (tfidf_vals >= -1e-5).all() and (tfidf_vals <= 1.0 + 1e-5).all(), \
            f"tfidf_cosine_name out of [0,1]: min={tfidf_vals.min():.4f} max={tfidf_vals.max():.4f}"
        print("✓ 6. tfidf_cosine_name values in [0, 1]")

        # ── 7: dense_cosine is zeros (patched) ───────────────────────────────
        dc_vals = result["dense_cosine"].to_numpy()
        assert (dc_vals == 0.0).all(), "Expected dense_cosine=0 (patched for test)"
        print("✓ 7. dense_cosine correctly patched to 0.0 (no model load)")

        # ── 8: 50k chunk streaming ────────────────────────────────────────────
        import random
        random.seed(42)
        words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa",
                 "123", "456", "789"]
        big_df = pl.DataFrame({
            "name1":    [" ".join(random.choices(words, k=3)) for _ in range(50_000)],
            "name2":    [" ".join(random.choices(words, k=3)) for _ in range(50_000)],
            "addr1":    [" ".join(random.choices(words, k=4)) for _ in range(50_000)],
            "addr2":    [" ".join(random.choices(words, k=4)) for _ in range(50_000)],
            "country1": ["us"] * 25_000 + ["india"] * 25_000,
            "country2": ["us"] * 20_000 + ["india"] * 20_000 + ["france"] * 10_000,
        })
        import time
        t0 = time.time()
        big_result = compute_features_chunked(big_df, chunk_size=50_000)
        elapsed = time.time() - t0
        assert len(big_result) == 50_000, f"Expected 50000 rows, got {len(big_result)}"
        assert all(c in big_result.columns for c in ALL_FEATURE_COLS), \
            "Missing feature columns in chunked result"
        print(f"✓ 8. 50k chunk: {len(big_result)} rows in {elapsed:.2f}s, all 28 cols present")

        # ── 9: I_num_conflict in compute_numeric_features (batch path) ────────
        num_df = pl.DataFrame({
            "name1": ["shop 12", "shop 12", "abc corp", "shop 12"],
            "name2": ["shop 13", "shop 12", "abc corp", "abc corp"],
            "addr1": ["main st", "main st", "", "main st"],
            "addr2": ["main st", "main st", "", "main st"],
        })
        num_result = compute_numeric_features(num_df)
        assert num_result["i_num_conflict"][0] == 1.0, "Shop 12 vs 13 batch: i_num_conflict should be 1.0"
        assert num_result["i_num_conflict"][1] == 0.0, "Shop 12 vs 12 batch: i_num_conflict should be 0.0"
        assert num_result["i_num_conflict"][2] == 0.0, "No numbers batch: i_num_conflict should be 0.0"
        assert num_result["i_num_conflict"][3] == 0.0, "One side batch: i_num_conflict should be 0.0"
        print("✓ 9. I_num_conflict batch path verified (4 cases)")

    finally:
        # Restore original function
        _g["_dense_cosine_batch"] = _orig_fn

    print()
    print("=" * 60)
    print("test_features() — ALL TESTS PASSED ✓  (28-dim features)")
    print("=" * 60)
