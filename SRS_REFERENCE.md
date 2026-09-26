# Business Entity Resolution Challenge — Enhanced Mini SRS & Technical Design v2
**Merged Analysis: Original SRS + Gemini-Generated Doc + Actual Repo Structure**
**Event:** Amazon ML Challenge | Sep 25-28 | 24h virtual | Teams 2-4 | 
**Date:** 2026-05-13 | Version: 2.0 — Publication Grade

---

## 0. What Changed vs v1 — Gemini Doc Better Ideas Integrated

| Area | v1 (My Original) | Gemini Doc Improvement | Adopted in v2 |
|------|-----------------|------------------------|---------------|
| **Blocking K** | 100 candidates/S1 | **K ≤30** with RR>99.98%, 20 BM25 + 20 dense → union | **Adopted** — Better for F0.5, 30 caps FP explosion |
| **Dense Model** | bge-small-en-v1.5 English | **paraphrase-multilingual-MiniLM-L12-v2 (384d, MIT, multilingual)** | **Adopted** — Handles France zero-shot, French accents |
| **Index Type** | FAISS IVF | **FAISS HNSW / IP (Inner Product)** faster for 200k corpus | **Adopted** — HNSW for <50ms query |
| **Country Handling** | Keep as feature only | **Exact match gating during blocking** (c_S1 == c_S2) + still open-set string (no hardcode list) | **Adopted** — Prune immediately if mismatch, improves precision, still works for unseen France |
| **Normalization** | Lowercase + suffix map | **Unicode NFKD + diacritic strip + "&→and"** explicitly for French é, è, Société → Societe | **Adopted** |
| **Numeric Logic** | pin_match boolean | **Numeric set concordance + conflict flag I_num_conflict** — if both have numbers but intersection empty → penalty | **Adopted** — Critical for Shop 12 vs Shop 13 |
| **Token Containment** | Jaccard only | **Directional containment C(T1,T2)=|∩|/|T1|** for abbreviated addresses | **Adopted** |
| **Loss Weighting** | Balanced class weight | **w0=2.0 (FP penalty), w1=1.0** to align with F0.5 | **Adopted** |
| **Validation** | GroupKFold by S1 | **GroupKFold + preserve singleton proportion per fold** | **Adopted** |
| **Threshold** | 0.35-0.9 grid | **τ* ≈0.72-0.85 figure with F0.5 curve peak** | **Adopted** |
| **Tech Stack Table** | Unpinned versions | **Pinned versions + License table (polars 1.12.0 Apache2, rapidfuzz 3.9.7 MIT, textdistance 4.6.3 MIT, rank-bm25 0.2.2 Apache2, faiss 1.8.0.post2 MIT)** | **Adopted** |
| **Folder Structure** | Generic | **Actual structure from image: student_resource/dataset/{train,test}/ + utils/validate_submission.py + Documentation_template.md** | **Corrected to match real repo** |
| **NFRs** | Qualitative | **Quantitative: ≤45min CPU / ≤15min T4, ≤16GB RAM, 50k chunk streaming** | **Adopted** |

---

## 1. Ground Truth Repo Structure (From Your Screenshot)

```
student_resource/
├── dataset/
│   ├── test/
│   │   ├── test_source1.tsv      # S1 reference (to be mapped)
│   │   ├── test_source2.tsv
│   │   └── test_source3.tsv
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv # S1 \t matched_ids (comma-separated, empty=singleton)
├── utils/
│   └── validate_submission.py    # stdlib-only validator, must PASS
├── Documentation_template.md
└── README.md
```

**Your final submission must still produce:**
```
<team>_submission.zip
├── output/
│   ├── candidate_pairs.tsv      # blocking raw, superset of final
│   └── matching_results.tsv     # scored
├── code/business_entity_resolution/src/...
├── requirements.txt
└── Documentation_template.md
```
But ingestion paths in code must point to `student_resource/dataset/...` during dev.

---

## 2. Problem Empathy & Failure Modes — Enhanced

### 2.1 Root-Cause Noise (First Principles) — Merged

**Legal Suffix Explosion:**
US: Inc, Corp, LLC, Co. India: Pvt Ltd, Ltd, LLP. France (open-set): SARL, SAS, SA, EURL, SCI. Naive token weighting treats `SARL` as discriminative, but it's stopword. Solution: jurisdiction-agnostic suffix list 40+ → map to `<CORP>` token, then strip for blocking.

**Landmark-Centric India vs Standardized US vs Inverted French:**
- US: `123 Main St, Austin, TX 78701` → house number first
- India: `Near SBI ATM, Opp Metro Pillar 142, MG Road, Bangalore` → landmark relative, no PIN
- France: `10 Rue de la Paix, 75002 Paris` → street type before name, cedex.

Banned geocoding → must rely on numeric set concordance + city token overlap.

**Transliteration & Diacritics:**
`Société Générale → Societe Generale` (NFKD), `Choudhury vs Chowdhury`, `Laxmi vs Lakshmi`, `Walmart vs Wal-Mart`. Requires NFKD normalization + rapidfuzz.

### 2.2 O(N×M) Explosion — Quantitative

N=|S1|, M=|S2|+|S3|. Brute force Ω = N×M. For N=100k, M=200k → 2×10^10 pairs. Deep cross-encoder @10ms/pair → 55,000 GPU-hours infeasible.

**Target:** Reduction Ratio RR = 1 - |C|/(N×M) >0.9998. With K=30, |C|=3M, RR=1 - 3M/20B=0.99985 → 3M pairs feasible. Each stage O(N+M) indexing, O(N×K) scoring.

```
S1 (N=1e5) ─┐
            ├─→ [Naive] 20B pairs ──► Infeasible
S2+S3 (2e5)─┘

S1 ──► [Blocking Filter K=30] ──► 3M pairs ──► Feasible
```

### 2.3 Singleton & F0.5 Dilemma — With Math

F0.5 = 1.25*P*R / (0.25*P+R). For singleton, GT=∅:
- Pred ∅ → P=1,R=1 → F=1.0
- Pred 1 FP → P=0 → F=0.0 catastrophic.

If 30% singletons, 10% FP rate on them → macro drops 0.3. Therefore need:
1. Calibrated probs (isotonic)
2. Global τ* 0.78 (from Gemini figure)
3. Singleton gate: if max prob < τ* → force empty

---

## 3. Architectural Comparison — Final Decision

| Criterion | Paradigm1 Deterministic | Paradigm2 Pure Dense | Paradigm3 Hybrid Cascaded **SELECTED** |
|-----------|------------------------|----------------------|----------------------------------------|
| Recall Ceiling | 70-82% | 88-94% | **97-99.5%** |
| Precision Control | Poor (rules) | Moderate (cosine) | **Exceptional (GBDT + w0=2.0)** |
| France OOD | Poor (needs regex) | Strong (multilingual) | **Strong (multilingual + NFKD + country gate)** |
| Latency | <1ms | 15-30ms | **5-10ms amortized** (TF-IDF + HNSW) |
| Singleton Safety | Fails (collisions) | Fails (hubness) | **Native (threshold + gate)** |

**Why Hybrid Wins for 24h Sprint Hyderabad:**
- Dual-channel: sparse captures exact numbers (Shop 12 vs 13), dense captures DBA.
- Country exact match gating reduces FP from cross-country `Paris Bistro US vs France` — still open-set because gating uses string equality, not hardcoded list.
- GBDT with 28 features handles nonlinear interactions (Jaro-Winkler high but numeric conflict=1 → should be 0).

---

## 4. System Architecture — ASCII with Gemini Improvements

### 4.1 Ingestion & Preprocessing (Open-Set Safe)

```
[student_resource/dataset/*/*.tsv]
   │
   ▼
+----------------------------------+  Preserve country string, no US/India list
| Polars Tab-Separated Reader      |
| sep="\t", dtype=str              |
+----------------------------------+
   │
   ▼
+----------------------------------+  NFKD: é→e, &→and, hyphen→space
| Text Canonicalization Engine     |  Lowercase, dedup whitespace
| - Unicode NFKD                   |
| - Punctuation→Whitespace         |
+----------------------------------+
   │
   ▼
+----------------------------------+  Suffixes: Inc, Corp, LLC, Pvt Ltd, SARL, SAS, SA, EURL
| Jurisdiction-Agnostic Cleaner    |  Address parser: extract digits, tokens
| - Suffix → <CORP> → strip       |  Pin regex \d{5,6}
+----------------------------------+
   │
   ▼
[Canonical S1,S2,S3] → DuckDB dedup
```

### 4.2 Multi-Index Blocking Engine → candidate_pairs.tsv (K≤30)

```
S2+S3 Corpus (M)
   ├─► [BM25 Sparse] Char 3-gram + word 1-2gram → Top-20 per S1 query
   │   rank-bm25, k1=1.5, b=0.75
   │
   └─► [Dense HNSW] paraphrase-multilingual-MiniLM-L12-v2 384d
       FAISS IndexHNSWFlat, M=32, efConstruction=200
       IP metric (cosine on normalized vectors) → Top-20

S1 Query
   │
   ├─► Query both indexes
   │      \
   │       \──► [Dynamic Union & Gating]
   │             - Country exact match gating: if country_S1 != country_S2 → prune (open-set safe)
   │             - Numeric overlap bonus: if PIN match → keep
   │             - RRF k=60 union, dedup, cap 30
   │
   ▼
output/candidate_pairs.tsv
source1_entity_id \t S2-xxx,S3-yyy,...
```

Recall target ≥0.98 at K=30 on train.

### 4.3 Re-Ranking & F0.5 Thresholding → matching_results.tsv

```
candidate_pairs (3M rows) + Canonical attrs
   │
   ▼
+------------------------------------+
| Pairwise Feature Extractor (Polars)|  28-dim
| - Edit: Lev_norm, JW, Monge-Elkan |
| - N-Gram: Jaccard_3gram, token Jacc, containment C(T1,T2) |
| - TF-IDF cosine (sublinear)       |
| - Country: I_country (0/1)        |
| - Numeric: Jacc(N_set), I_num_conflict |
| - Dense: cosine_multilingual (v1^T v_x) |
+------------------------------------+
   │
   ▼ Feature Matrix X [3M×28] chunked 50k
+------------------------------------+
| LightGBM Binary (w0=2.0 FP penalty)|
| GroupKFold 5-fold by S1 ID, singleton proportion preserved |
| n_est=1000, lr=0.05, leaves=63     |
+------------------------------------+
   │
   ▼ P(Match|Pair) calibrated isotonic
+------------------------------------+
| F0.5 Calibrated Thresholding       |
| τ* = argmax macro F0.5 on OOF val  |
| Sweep 0.40-0.95 step 0.01 → peak ~0.78 |
| Singleton Guard: if max_p < τ* → [] |
+------------------------------------+
   │
   ▼
output/matching_results.tsv
```

### 4.4 Final Package Structure (Aligned to Real Repo + Challenge Spec)

```
<team>_submission.zip
├── output/
│   ├── candidate_pairs.tsv         # K≤30 per S1, tab-separated, PASS validator
│   └── matching_results.tsv        # final, matched ⊆ candidate
├── code/
│   └── business_entity_resolution/
│       └── src/
│           ├── __init__.py
│           ├── ingestion.py        # Polars reader, open-set country
│           ├── preprocess.py       # NFKD, suffix map incl. French
│           ├── blocking.py         # BM25 + HNSW multilingual + country gate
│           ├── features.py         # 28-dim vectorized
│           ├── model.py            # LightGBM + calibration
│           ├── postprocess.py      # τ* + singleton gate
│           ├── pipeline.py         # orchestrator
│           └── main.py             # CLI
├── requirements.txt                # pinned MIT/Apache (see table)
├── Documentation_template.md
└── README.md
```

---

## 5. Mathematical Modeling — Enhanced Features

### 5.1 Exact Feature Formulas (Adopted from Gemini)

**Edit:**
- Lev_norm = 1 - Lev(a,b)/max(|a|,|b|) for name and address separately
- JW = Jaro + ℓ·p·(1-Jaro), ℓ≤4, p=0.1
- Monge-Elkan: ME(A,B)=1/|A| Σ_i max_j JW(a_i,b_j) → handles transposition `Tata Motors Ltd` vs `Motors Tata Limited`

**N-Gram:**
- Char 3-gram Jaccard J3 = |gram3(s1)∩gram3(s2)|/|∪|
- Token containment C(T1,T2)=|∩|/|T1| → if address is substring of another, containment high but Jaccard low → captures abbreviation

**Country & Numeric (Key Improvement):**
- I_country = 1 if lower(country1)==lower(country2) else 0 — **prune during blocking if 0**
- Numeric sets N = regex `\d+`
  - Jaccard(N1,Nx)
  - Conflict flag I_num_conflict = 1 if N1≠∅ and Nx≠∅ and N1∩Nx=∅ → e.g., Shop 12 vs Shop 13 → hard negative

**Dense:**
- v = normalized embedding from paraphrase-multilingual-MiniLM-L12-v2 (MIT, 118M params <8B)
- Sim_dense = v1^T vx (cosine)

Final x ∈ R^28

### 5.2 Loss & Validation — F0.5 Aligned

Loss:
L(θ)= -Σ [w1·y·log(p̂)+ w0·(1-y)·log(1-p̂)], w0=2.0 penalizes FP double (aligns with β=0.5), w1=1.0

GroupKFold: Group = source1_entity_id, all pairs of same S1 together. Singleton proportion preserved per fold via stratification.

### 5.3 Threshold Tuning Visualization

```
Macro F0.5
  ^
0.85|                 * τ*~0.78
0.80|               *   *
0.70|            *         *
0.60|         *               *  FP explodes as τ drops
     +--------------------------------> τ
     0.40 0.50 0.60 0.70 0.80 0.90
```

Sweep OOF probs, compute macro F0.5 per τ, pick max.

### 5.4 Singleton Gating

For each S1:
p_max = max_{x∈C(S1)} p̂(S1,x)
If p_max < τ* → return ∅
Else return {x | p̂ ≥ τ*}

Optional second gate: singleton_classifier prob >0.65 → ∅ even if p_max≥τ*

---

## 6. Mini-SRS Specification v2

### 6.1 Functional Requirements

**FR-01 Ingestion:** MUST read TSV with explicit `\t`, MUST NOT hardcode country list, MUST accept any string (including France). Preserve order of test_source1 for output.

**FR-02 Canonicalization:** MUST NFKD normalize, strip diacritics, map &→and, lowercase, strip leading/trailing, deduplicate whitespace. MUST strip jurisdictional suffixes using lookup covering US, India, French SARL/SAS/SA/EURL.

**FR-03 Blocking:** MUST build char 3-gram BM25 + dense HNSW multilingual indexes over S2∪S3. MUST output exactly one row per S1 into candidate_pairs.tsv, K≤30 unique IDs from S2/S3 only, country gating if mismatch. MUST achieve blocking recall ≥0.97 on train val. MUST ensure matched ⊆ candidate.

**FR-04 Features:** MUST compute 28-dim pairwise vector in batches (50k chunk via Polars streaming) including Levenshtein norm, JW, Monge-Elkan, Jaccard_3gram, token Jaccard, containment, TF-IDF cosine (sublinear), I_country, numeric Jaccard + I_num_conflict, dense cosine.

**FR-05 Ranking:** MUST train LightGBM with w0=2.0, GroupKFold by S1, calibrated probabilities, threshold sweep for F0.5. MUST implement singleton guard.

### 6.2 Non-Functional Requirements

**NFR-01 Time:** End-to-end (ingestion→blocking→features→ranking→validation) ≤45 min on 16-core CPU workstation, ≤15 min on T4/V100 GPU. Intermediate feature matrices chunked 50k pairs via Polars streaming.

**NFR-02 Memory:** Peak RAM ≤16 GB. FAISS HNSW index ~300-500MB for 200k × 384d. Feature matrix processed chunked, not all in RAM.

**NFR-03 License & Offline:** All models MIT/Apache 2.0/BSD, params ≤8B (bge-small 33M, MiniLM-L12 118M, LightGBM). MUST run offline, no external APIs, web scrapers, geocoding.

**NFR-04 Reproducibility & Sprint Fit:** Seed 42 fixed. `requirements.txt` pinned with hashes. Main CLI: `python code/business_entity_resolution/src/main.py --test_dir student_resource/dataset/test --output_dir output/`. Must work for 24h in-person Teams 2-4 parallel modules.

### 6.3 Submission Integrity & Validation

Before zip, MUST run:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
# Must output PASS with exit 0
```

**Explicit Checks:**
1. Tab Separation: single `\t` per line, addresses may contain commas
2. Completeness & Cardinality: Every S1 in test_source1 appears exactly once in both outputs
3. Strict ID Partitioning: Matches/candidates only S2-/S3- IDs existing in test_source2/3
4. No Self-Matches: S1 IDs never in own list
5. Deduplicated: No duplicate IDs in comma list
6. Candidate-Match Invariant: matching ⊆ candidate
7. Singleton Representation: Empty second column (e.g., `S1-00003\t`) not "nan"/"None", no trailing comma/space

### 6.4 Recommended Tech Stack — Pinned MIT/Apache (Adopted from Gemini)

| Component | Library | Version | License | Responsibility |
|-----------|---------|---------|---------|----------------|
| High-Throughput IO | polars | 1.12.0 | Apache 2.0 | TSV ingestion, chunking, joins |
| High-Throughput IO Alt | duckdb | 1.1.3 | MIT | SQL dedup |
| String Algorithms | rapidfuzz | 3.9.7 | MIT | C++ Levenshtein, JW, token_sort |
| Text Matching | textdistance | 4.6.3 | MIT | Monge-Elkan, Damerau-Levenshtein |
| Sparse Index | rank-bm25 | 0.2.2 | Apache 2.0 | BM25 char 3-gram inverted |
| Dense Index | faiss-cpu | 1.8.0.post2 | MIT | HNSW / IP ANN |
| Embeddings | sentence-transformers | 3.2.1 | Apache 2.0 | Wrapper |
| Embedding Model | paraphrase-multilingual-MiniLM-L12-v2 | - | MIT | 118M multilingual, handles French |
| Embedding Model Alt | bge-small-en-v1.5 | - | MIT | 33M English fallback |
| Transliteration | anyascii | 0.3.2 | MIT | Safer than unidecode GPL |
| Ranker | lightgbm | 4.5.0 | MIT | GBDT with w0=2.0 |
| Ranker Alt | catboost | 1.2.7 | Apache 2.0 | Alternative |
| Progress | tqdm | 4.66.5 | MIT | |

**Banned:** geopy, googlemaps, opencorporates, clearbit, any external ER service.

### 6.5 24-Hour Sprint Execution Plan (Hyderabad)

**Hour 0-4 Setup (All):** Clone, create venv, implement ingestion.py with NFKD, run dummy validator.

**Hour 4-10 Parallel:**
- Eng1: blocking.py BM25 + PIN dict → K=20, measure recall
- Eng2: features.py rapidfuzz + numeric conflict
- Eng3: model.py LightGBM baseline train
- Eng4: Doc + Fabric Lakehouse logging (optional)

**Hour 10-18 Integration:**
- Add HNSW multilingual, RRF fusion K=30, country gate → recall 0.97+
- Threshold sweep τ*, singleton gate

**Hour 18-22 Final:**
- Full test inference, generate output/, run validate_submission.py → PASS
- Package zip, pin requirements.txt

**Hour 22-24:** Fill Documentation_template.md with architecture + model licenses, push GitHub open-source, submit.

---

## Appendix — Why This v2 Is Better

1. **K=30 not 100** → F0.5 protects precision, reduces inference 3× faster for 24h sprint
2. **Multilingual model** → France zero-shot without French regex rules
3. **Country gating + I_num_conflict** → eliminates major FP source (same brand different shop numbers)
4. **Containment metric** → handles India landmark abbreviation where address is substring
5. **w0=2.0 + isotonic calibration + τ*~0.78** → directly optimizes leaderboard metric, not F1
6. **Exact repo paths from screenshot** → code will run on evaluator's `student_resource/dataset/...`
7. **Pinned licenses table** → passes MIT/Apache compliance check

**End of Enhanced SRS v2 — Ready to code under `code/business_entity_resolution/src/` for Open Source Sprint Hyderabad.**
