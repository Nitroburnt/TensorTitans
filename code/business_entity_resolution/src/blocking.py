"""
blocking.py — Dual-channel blocking engine for Business Entity Resolution.

Architecture (per user scale note: ~10M corpus):
  BM25 channel  : rank-bm25 BM25Okapi, char 3-gram + word 1-2gram tokens
  Dense channel : FAISS IndexHNSWFlat (country-sharded), L2-normalized embeddings
                  M=32, efConstruction=200, IP metric → inner product ≡ cosine similarity
                  Model: paraphrase-multilingual-MiniLM-L12-v2 MIT 118M 384-d
  Fusion        : Reciprocal Rank Fusion k=60, union of both channels
  Gate          : country exact match (string equality, open-set safe)
  Cap           : K ≤ 30 per S1 entity

Country sharding rationale:
  10M corpus × single HNSW → large but feasible with HNSW's O(log N) query.
  Splitting by lower(country) keeps each shard ≤ 5M rows and makes country
  gating implicit at query time (open-set safe — no hardcoded country list).

Outputs:
  candidate_pairs.tsv : source1_entity_id \\t candidate_entity_ids (comma-joined)
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42
BM25_TOP_K = 20          # candidates per S1 from sparse channel
DENSE_TOP_K = 20         # candidates per S1 from dense channel
RRF_K = 60               # RRF smoothing constant
FINAL_K = 30             # max candidates per S1 in output
EMBED_BATCH = 512        # sentences per encode() call
HNSW_M = 32              # HNSW graph connectivity (higher M → better recall, more RAM)
HNSW_EF_CONSTRUCTION = 200  # HNSW build-time beam width (higher → better quality graph)
HNSW_EF_SEARCH = 64     # HNSW query-time beam width (higher → better recall, slower)
DENSE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ── Tokenizer for BM25 ────────────────────────────────────────────────────────

_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def _char_ngrams(text: str, n: int) -> list[str]:
    """All character n-grams of a string (no padding)."""
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def bm25_tokenize(text: str) -> list[str]:
    """Tokenizer combining char 3/4/5-grams + word unigrams + word bigrams.

    Char 3-5 grams catch transliteration variants (Choudhury/Chowdhury)
    and partial suffix overlaps at multiple granularities.
    Word unigrams/bigrams anchor on salient tokens.
    """
    text = _NON_ALNUM.sub(" ", text.lower()).strip()
    if not text:
        return ["__empty__"]
    words = text.split()
    tokens: list[str] = []
    # char 3-grams, 4-grams, 5-grams of the whole cleaned string (no spaces)
    compact = "".join(words)
    for n in (3, 4, 5):
        tokens.extend(_char_ngrams(compact, n))
    # word unigrams
    tokens.extend(words)
    # word bigrams
    tokens.extend(f"{a}_{b}" for a, b in zip(words, words[1:]))
    return tokens if tokens else ["__empty__"]


# ── BM25 shard ────────────────────────────────────────────────────────────────

def _identity_tokenizer(text: str) -> list[str]:
    """Pass-through tokenizer for sklearn (tokens already joined by space)."""
    return text.split()


class BM25Shard:
    """Inverted-index BM25 over one country shard.

    Uses BM25Okapi scoring with an inverted index for O(posting_list) queries
    instead of O(N) dense scoring. At 6M docs per country shard:
    - Build: ~60-90s (tokenize + invert)
    - Query: ~0.1-2ms per query (only non-zero posting lists scored)

    The inverted index maps each token → list of (doc_idx, term_freq).
    BM25 score = Σ_t IDF(t) · (tf·(k1+1))/(tf + k1·(1-b+b·dl/avgdl))
    with k1=1.5, b=0.75 per BM25Okapi defaults.
    """

    K1: float = 1.5
    B: float = 0.75

    def __init__(self, ids: list[str], texts: list[str]) -> None:
        self.ids = ids
        N = len(ids)
        if N == 0:
            self._idf: dict[str, float] = {}
            self._inv: dict[str, list[tuple[int, float]]] = {}
            self._dl_norm: np.ndarray = np.zeros(0, dtype=np.float32)
            return

        # Tokenize
        tokenized: list[list[str]] = [bm25_tokenize(t) for t in texts]

        # Document lengths
        dls = np.array([len(doc) for doc in tokenized], dtype=np.float32)
        avgdl = float(dls.mean())
        # Precompute BM25 denominator term: k1*(1-b+b*dl/avgdl)
        denom_base = self.K1 * (1.0 - self.B + self.B * dls / max(avgdl, 1.0))
        self._dl_norm = denom_base  # shape (N,)

        # Build inverted index: token → [(doc_idx, tf), ...]
        # Also count DF for IDF
        inv_raw: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, doc_tokens in enumerate(tokenized):
            tf_map: dict[str, int] = defaultdict(int)
            for tok in doc_tokens:
                tf_map[tok] += 1
            for tok, tf in tf_map.items():
                inv_raw[tok].append((doc_idx, tf))

        # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)  [BM25+ variant, always positive]
        self._idf = {}
        for tok, postings in inv_raw.items():
            df = len(postings)
            self._idf[tok] = float(np.log((N - df + 0.5) / (df + 0.5) + 1.0))

        # Store precomputed TF-weighted postings as float32 for fast scoring
        # Posting: (doc_idx, tf_f32) — we multiply IDF at query time
        self._inv: dict[str, list[tuple[int, float]]] = {
            tok: [(di, float(tf)) for di, tf in postings]
            for tok, postings in inv_raw.items()
        }

    def query(self, text: str, top_k: int = BM25_TOP_K) -> list[tuple[str, int]]:
        """Return [(entity_id, rank_1based), ...] for top_k results."""
        if not self.ids:
            return []
        q_tokens = bm25_tokenize(text)
        # Accumulate BM25 scores only for documents in posting lists
        scores: dict[int, float] = {}
        k1 = self.K1
        for tok in set(q_tokens):  # unique query tokens
            idf = self._idf.get(tok, 0.0)
            if idf == 0.0:
                continue
            postings = self._inv.get(tok, [])
            for doc_idx, tf in postings:
                # BM25 TF component: tf*(k1+1) / (tf + denom_base[doc_idx])
                tf_score = tf * (k1 + 1.0) / (tf + float(self._dl_norm[doc_idx]))
                if doc_idx in scores:
                    scores[doc_idx] += idf * tf_score
                else:
                    scores[doc_idx] = idf * tf_score

        if not scores:
            # no token overlap — return first top_k as fallback
            n = min(top_k, len(self.ids))
            return [(self.ids[i], i + 1) for i in range(n)]

        n = min(top_k, len(scores))
        top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(self.ids[doc_idx], rank + 1) for rank, (doc_idx, _) in enumerate(top_items)]

    def query_batch(self, texts: list[str], top_k: int = BM25_TOP_K) -> list[list[tuple[str, int]]]:
        """Batch query — iterates per-text but reuses shard data structures."""
        return [self.query(t, top_k=top_k) for t in texts]


# ── BM25-only standalone blocking ────────────────────────────────────────────

def run_bm25_blocking(
    s1_df: pl.DataFrame,
    corpus_df: pl.DataFrame,
    top_k: int = BM25_TOP_K,
) -> dict[str, list[tuple[str, int]]]:
    """Run BM25-only blocking (no dense channel).

    Country-exact-match gating is applied: each S1 entity is queried only
    against the corpus shard matching its lower(country_clean). Open-set safe
    — no hardcoded country list; any unseen country (e.g. France) gets its own
    shard automatically.

    Parameters
    ----------
    s1_df : pl.DataFrame
        Preprocessed S1 with columns: entity_id, name_clean, address_clean,
        country_clean.
    corpus_df : pl.DataFrame
        Preprocessed S2∪S3 with same columns (from build_corpus + preprocess).
    top_k : int
        Number of BM25 candidates per S1 entity (default 20).

    Returns
    -------
    dict[str, list[tuple[str, int]]]
        { s1_entity_id : [(candidate_entity_id, rank_1based), ...] }
        Entities whose country has no corpus shard return an empty list.
    """
    # Build one BM25Shard per country from corpus
    bm25_shards: dict[str, BM25Shard] = {}
    corpus_countries = corpus_df["country_clean"].unique().to_list()

    for country in sorted(corpus_countries):
        shard_df = corpus_df.filter(pl.col("country_clean") == country)
        ids = shard_df["entity_id"].to_list()
        texts = [
            f"{n} {a}"
            for n, a in zip(shard_df["name_clean"].to_list(),
                            shard_df["address_clean"].to_list())
        ]
        bm25_shards[country] = BM25Shard(ids, texts)
        print(f"[bm25] Built shard [{country}] size={len(ids):,}")

    # Query each S1 entity against its matching country shard
    results: dict[str, list[tuple[str, int]]] = {}
    s1_countries = s1_df["country_clean"].unique().to_list()

    for country in sorted(s1_countries):
        shard_s1 = s1_df.filter(pl.col("country_clean") == country)
        ids_s1 = shard_s1["entity_id"].to_list()
        texts_s1 = [
            f"{n} {a}"
            for n, a in zip(shard_s1["name_clean"].to_list(),
                            shard_s1["address_clean"].to_list())
        ]

        if country not in bm25_shards:
            print(f"[bm25] WARNING: no corpus shard for country '{country}', skipping {len(ids_s1)} S1 entities")
            for s1id in ids_s1:
                results[s1id] = []
            continue

        shard = bm25_shards[country]
        print(f"[bm25] Querying [{country}] S1 count={len(ids_s1):,}")
        BATCH = 512
        for start in tqdm(range(0, len(ids_s1), BATCH),
                          desc=f"  BM25 {country}",
                          unit="batch",
                          disable=len(ids_s1) < 500):
            end = min(start + BATCH, len(ids_s1))
            batch_ids = ids_s1[start:end]
            batch_texts = texts_s1[start:end]
            batch_hits = shard.query_batch(batch_texts, top_k=top_k)
            for s1id, hits in zip(batch_ids, batch_hits):
                results[s1id] = hits

    return results


# ── Dense HNSW shard ──────────────────────────────────────────────────────────

class DenseShard:
    """FAISS IndexHNSWFlat (cosine via IP on L2-normalized vectors) for one country.

    Index parameters per SRS v2:
      M=32              — graph connectivity; higher M → better recall, more RAM/build time
      efConstruction=200 — beam width during graph construction; higher → better quality
      efSearch=64       — beam width at query time; higher → better recall, slower queries
      METRIC_INNER_PRODUCT — IP on L2-normalized vectors == cosine similarity
    """

    def __init__(self, ids: list[str], embeddings: np.ndarray) -> None:
        """
        Parameters
        ----------
        ids        : corpus entity_ids in row order
        embeddings : float32 array shape (N, d), already L2-normalized
        """
        self.ids = ids
        d = embeddings.shape[1]
        # IndexHNSWFlat does not require a train() call — add directly
        self.index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self.index.hnsw.efSearch = HNSW_EF_SEARCH
        self.index.add(embeddings)

    def query(self, embeddings: np.ndarray, top_k: int = DENSE_TOP_K) -> list[list[tuple[str, int]]]:
        """Batch query. Returns list[list[(entity_id, rank_1based)]] one per query."""
        n = min(top_k, len(self.ids))
        D, I = self.index.search(embeddings, n)
        results = []
        for row in I:
            results.append(
                [(self.ids[i], rank + 1) for rank, i in enumerate(row) if i >= 0]
            )
        return results


# ── RRF fusion ────────────────────────────────────────────────────────────────

def rrf_fuse(
    sparse_hits: list[tuple[str, int]],
    dense_hits: list[tuple[str, int]],
    k: int = RRF_K,
    final_k: int = FINAL_K,
) -> list[str]:
    """Reciprocal Rank Fusion of two ranked lists.

    score(d) = Σ  1/(k + rank(d, channel))
    Returns top final_k entity_ids, best score first.
    """
    scores: dict[str, float] = defaultdict(float)
    for eid, rank in sparse_hits:
        scores[eid] += 1.0 / (k + rank)
    for eid, rank in dense_hits:
        scores[eid] += 1.0 / (k + rank)
    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return ranked[:final_k]


# ── BlockingEngine ────────────────────────────────────────────────────────────

class BlockingEngine:
    """Full dual-channel blocking pipeline.

    Usage
    -----
    engine = BlockingEngine()
    engine.build(corpus_df)          # corpus_df from preprocess_dataframe
    candidates = engine.query_all(s1_df)   # returns dict s1_id → [s2/s3_ids]
    engine.write_candidate_pairs(candidates, s1_df, output_path)

    bm25_only=True skips the sentence-transformer model and FAISS index entirely,
    using only BM25 at K=FINAL_K. Reduces peak RAM by ~15 GB on a 10M corpus.
    """

    def __init__(self, model_name: str = DENSE_MODEL, bm25_only: bool = False) -> None:
        self.bm25_only = bm25_only
        if not bm25_only:
            print(f"[blocking] Loading sentence encoder: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.dim: int = self.model.get_sentence_embedding_dimension()
        else:
            print("[blocking] BM25-only mode (no dense encoder)")
            self.model = None
            self.dim = 0
        # per-country shards
        self._bm25_shards: dict[str, BM25Shard] = {}
        self._dense_shards: dict[str, DenseShard] = {}
        self._corpus_df: pl.DataFrame | None = None

    # ── build ──────────────────────────────────────────────────────────────

    def build(self, corpus_df: pl.DataFrame) -> None:
        """Index the corpus (preprocessed S2∪S3 DataFrame).

        Expected columns: entity_id, name_clean, address_clean, country_clean

        In bm25_only=True mode, defers BM25 index building entirely — no shards
        are stored here.  Instead, the corpus DataFrame reference is saved so
        query_all() can stream one country shard at a time (build→query→discard),
        keeping peak RAM to a single country's BM25 index rather than all of them.

        In dual-channel mode, builds one BM25Shard + one DenseShard per country.
        """
        if self.bm25_only:
            # Defer BM25 index building to query_all() which will stream per country.
            # Storing the full corpus ref here is fine — it's the same Polars
            # DataFrame that pipeline.py already has in RAM; no duplication.
            self._corpus_df = corpus_df
            countries = corpus_df["country_clean"].unique().to_list()
            print(f"[blocking] BM25-only deferred mode: corpus={len(corpus_df):,}  "
                  f"Countries: {sorted(countries)}")
            return

        # ── Dual-channel: build BM25 + FAISS shards per country ──────────────
        self._corpus_df = corpus_df
        countries = corpus_df["country_clean"].unique().to_list()
        print(f"[blocking] Corpus size: {len(corpus_df):,}  Countries: {sorted(countries)}")

        for country in sorted(countries):
            shard_df = corpus_df.filter(pl.col("country_clean") == country)
            ids = shard_df["entity_id"].to_list()
            texts = [
                f"{n} {a}"
                for n, a in zip(shard_df["name_clean"].to_list(),
                                shard_df["address_clean"].to_list())
            ]
            n = len(ids)
            print(f"  [{country}] shard size: {n:,}")

            # ── BM25 ──────────────────────────────────────────────────────
            t0 = time.time()
            self._bm25_shards[country] = BM25Shard(ids, texts)
            print(f"    BM25 indexed in {time.time()-t0:.1f}s")

            # ── Dense embeddings ──────────────────────────────────────────
            t0 = time.time()
            embs = self._encode_texts(texts)
            print(f"    Encoded {n:,} in {time.time()-t0:.1f}s")

            t0 = time.time()
            self._dense_shards[country] = DenseShard(ids, embs)
            del embs  # free immediately after indexing
            import gc; gc.collect()
            print(f"    HNSW built M={HNSW_M} efC={HNSW_EF_CONSTRUCTION} in {time.time()-t0:.1f}s")

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode list of strings → L2-normalized float32 array (N, d)."""
        embs = self.model.encode(
            texts,
            batch_size=EMBED_BATCH,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-norm in-place → IP == cosine
        ).astype(np.float32)
        return embs

    # ── streaming BM25 build+query (bm25_only mode) ───────────────────────

    def _build_query_bm25_streaming(
        self,
        corpus_df: pl.DataFrame,
        s1_df: pl.DataFrame,
        batch_size: int = 512,
    ) -> dict[str, list[str]]:
        """Build BM25 shard per country, query S1 immediately, then discard.

        Peak RAM = one country's BM25 inverted index at a time (not all).
        Open-set safe: any unseen country in corpus gets its own shard.
        S1 entities whose country has no corpus shard get an empty candidate list.
        """
        import gc
        candidates: dict[str, list[str]] = {}

        # Pre-group S1 queries by country (tiny — S1 is small)
        s1_by_country: dict[str, tuple[list[str], list[str]]] = {}
        for country in s1_df["country_clean"].unique().to_list():
            shard = s1_df.filter(pl.col("country_clean") == country)
            s1_by_country[country] = (
                shard["entity_id"].to_list(),
                [f"{n} {a}" for n, a in zip(
                    shard["name_clean"].to_list(),
                    shard["address_clean"].to_list(),
                )],
            )

        # Process corpus one country at a time
        corpus_countries = corpus_df["country_clean"].unique().to_list()
        print(f"[blocking] Corpus size: {len(corpus_df):,}  Countries: {sorted(corpus_countries)}")

        for country in sorted(corpus_countries):
            shard_df = corpus_df.filter(pl.col("country_clean") == country)
            ids = shard_df["entity_id"].to_list()
            texts = [f"{n} {a}" for n, a in zip(
                shard_df["name_clean"].to_list(),
                shard_df["address_clean"].to_list(),
            )]
            del shard_df  # free filter result
            print(f"  [{country}] corpus shard: {len(ids):,}")

            t0 = time.time()
            bm25 = BM25Shard(ids, texts)
            del ids, texts
            gc.collect()
            print(f"    BM25 indexed in {time.time()-t0:.1f}s")

            if country not in s1_by_country:
                # No S1 queries for this country — still discard index
                del bm25
                gc.collect()
                continue

            ids_s1, texts_s1 = s1_by_country[country]
            n = len(ids_s1)
            print(f"    Querying {n:,} S1 entities...")

            for start in tqdm(range(0, n, batch_size),
                              desc=f"  {country}",
                              unit="batch",
                              disable=n < 1000):
                end = min(start + batch_size, n)
                hits_batch = bm25.query_batch(texts_s1[start:end], top_k=FINAL_K)
                for i, s1id in enumerate(ids_s1[start:end]):
                    candidates[s1id] = [eid for eid, _ in hits_batch[i]]

            del bm25
            gc.collect()

        # S1 entities whose country had no corpus shard → empty list
        for ids_s1, _ in s1_by_country.values():
            for s1id in ids_s1:
                if s1id not in candidates:
                    candidates[s1id] = []

        return candidates

    # ── query_all ─────────────────────────────────────────────────────────

    def query_all(
        self,
        s1_df: pl.DataFrame,
        batch_size: int = 256,
        corpus_df: pl.DataFrame | None = None,
    ) -> dict[str, list[str]]:
        """Query both channels for all S1 entities.

        Parameters
        ----------
        s1_df      : preprocessed S1 DataFrame
        batch_size : batch size for dense encoding (ignored in bm25_only mode)
        corpus_df  : if provided AND bm25_only=True, uses streaming build+query
                     (one country at a time) instead of pre-built shards.
                     This is the low-RAM path for large corpora.

        Returns dict { s1_entity_id : [candidate_entity_ids, ...] }
        Candidates are already country-gated and capped at FINAL_K.
        """
        # ── bm25_only streaming path (low RAM) ────────────────────────────
        if self.bm25_only:
            # Prefer corpus_df passed at query time; fall back to stored ref.
            _corpus = corpus_df if corpus_df is not None else self._corpus_df
            if _corpus is not None:
                return self._build_query_bm25_streaming(_corpus, s1_df, batch_size=512)
            # Legacy fallback: pre-built shards in self._bm25_shards
            candidates: dict[str, list[str]] = {}
            countries = s1_df["country_clean"].unique().to_list()
            for country in sorted(countries):
                shard_s1 = s1_df.filter(pl.col("country_clean") == country)
                ids_s1 = shard_s1["entity_id"].to_list()
                texts_s1 = [
                    f"{n} {a}"
                    for n, a in zip(shard_s1["name_clean"].to_list(),
                                    shard_s1["address_clean"].to_list())
                ]
                n = len(ids_s1)
                print(f"[blocking] Querying [{country}] S1 count={n:,}")
                if country not in self._bm25_shards:
                    print(f"  WARNING: no corpus shard for country '{country}', returning empty")
                    for s1id in ids_s1:
                        candidates[s1id] = []
                    continue
                bm25_shard = self._bm25_shards[country]
                for start in tqdm(range(0, n, 512),
                                  desc=f"  {country} BM25",
                                  unit="batch",
                                  disable=n < 1000):
                    end = min(start + 512, n)
                    hits_batch = bm25_shard.query_batch(texts_s1[start:end], top_k=FINAL_K)
                    for i, s1id in enumerate(ids_s1[start:end]):
                        candidates[s1id] = [eid for eid, _ in hits_batch[i]]
            return candidates

        # ── Dual-channel path ─────────────────────────────────────────────
        candidates: dict[str, list[str]] = {}
        countries = s1_df["country_clean"].unique().to_list()

        for country in sorted(countries):
            shard_s1 = s1_df.filter(pl.col("country_clean") == country)
            ids_s1 = shard_s1["entity_id"].to_list()
            texts_s1 = [
                f"{n} {a}"
                for n, a in zip(shard_s1["name_clean"].to_list(),
                                shard_s1["address_clean"].to_list())
            ]
            n = len(ids_s1)
            print(f"[blocking] Querying [{country}] S1 count={n:,}")

            # skip if no corpus shard for this country
            if country not in self._bm25_shards:
                print(f"  WARNING: no corpus shard for country '{country}', returning empty")
                for s1id in ids_s1:
                    candidates[s1id] = []
                continue

            bm25_shard = self._bm25_shards[country]
            dense_shard = self._dense_shards[country]

            # Encode all S1 queries for this country up-front
            t0 = time.time()
            q_embs = self._encode_texts(texts_s1)
            print(f"  Encoded {n:,} S1 queries in {time.time()-t0:.1f}s")

            for start in tqdm(range(0, n, batch_size),
                              desc=f"  {country} blocking",
                              unit="batch",
                              disable=n < 1000):
                end = min(start + batch_size, n)
                batch_ids = ids_s1[start:end]
                batch_texts = texts_s1[start:end]
                batch_embs = q_embs[start:end]

                # Dense hits (batched)
                dense_hits_batch = dense_shard.query(batch_embs, top_k=DENSE_TOP_K)
                # BM25 hits (batched)
                sparse_hits_batch = bm25_shard.query_batch(batch_texts, top_k=BM25_TOP_K)

                for i, s1id in enumerate(batch_ids):
                    fused = rrf_fuse(sparse_hits_batch[i], dense_hits_batch[i], k=RRF_K, final_k=FINAL_K)
                    candidates[s1id] = fused

        return candidates

    # ── write output ──────────────────────────────────────────────────────

    @staticmethod
    def write_candidate_pairs(
        candidates: dict[str, list[str]],
        s1_df: pl.DataFrame,
        output_path: str | Path,
    ) -> None:
        """Write candidate_pairs.tsv preserving s1_df row order.

        Format:
          source1_entity_id \\t candidate_entity_ids
          (candidate_entity_ids = comma-separated S2/S3 IDs, "" if empty)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        s1_ids_ordered = s1_df["entity_id"].to_list()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1id in s1_ids_ordered:
                cands = candidates.get(s1id, [])
                f.write(f"{s1id}\t{','.join(cands)}\n")

        total_cands = sum(len(v) for v in candidates.values())
        print(f"[blocking] Wrote {len(s1_ids_ordered):,} rows to {output_path}")
        print(f"[blocking] Avg candidates/S1: {total_cands/max(len(candidates),1):.1f}")


# ── Blocking recall evaluator (for gate check) ────────────────────────────────

def evaluate_blocking_recall(
    candidates: dict[str, list[str]],
    ground_truth: pl.DataFrame,
    label: str = "",
) -> float:
    """Compute blocking recall: |GT ∩ candidates| / |GT| macro-averaged over S1.

    Only evaluates S1 entities that appear in ground_truth AND have GT matches
    (non-empty matched_entity_ids). Singletons are excluded (they don't affect
    blocking recall since there's nothing to retrieve).

    Parameters
    ----------
    candidates    : dict s1_id → list of candidate s2/s3 ids
    ground_truth  : pl.DataFrame with columns [source1_entity_id, matched_entity_ids]
    label         : optional label for print output

    Returns
    -------
    macro-averaged recall (float in [0, 1])
    """
    recalls: list[float] = []
    missed_examples: list[str] = []

    for row in ground_truth.iter_rows(named=True):
        s1id = row["source1_entity_id"]
        gt_str = row["matched_entity_ids"].strip()
        if not gt_str:
            continue  # singleton — skip
        gt_set = set(gt_str.split(","))
        cands_set = set(candidates.get(s1id, []))
        r = len(gt_set & cands_set) / len(gt_set)
        recalls.append(r)
        if r < 1.0 and len(missed_examples) < 3:
            missed = gt_set - cands_set
            missed_examples.append(f"  {s1id}: missed {list(missed)[:2]}")

    macro_recall = float(np.mean(recalls)) if recalls else 0.0
    tag = f"[{label}] " if label else ""
    print(f"{tag}Blocking recall: {macro_recall:.4f}  (over {len(recalls)} non-singleton S1s)")
    if missed_examples:
        print(f"{tag}Sample misses:")
        for ex in missed_examples:
            print(ex)
    return macro_recall
