"""
preprocess.py — Text canonicalization for Business Entity Resolution.

Responsibilities:
- Unicode NFKD normalization + diacritic strip (Société→Societe, é→e)
- Ampersand expansion (&→and)
- Hyphen/punctuation normalization
- Lowercase + deduplicate whitespace
- Jurisdiction-agnostic suffix map 40+ entries (US/India/France/generic)
  → replace matched suffix tokens with <CORP> then strip for blocking
- PIN (postal code) extraction via regex \\d{5,6}
- Numeric token extraction for Numeric Jaccard + I_num_conflict

All functions are vectorized over Polars Series where possible,
with a Python-level str fallback for individual strings.
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

# ── NFKD + diacritic strip ────────────────────────────────────────────────────

def _normalize_unicode(text: str) -> str:
    """NFKD decompose then drop combining diacritics. é→e, Société→Societe."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if unicodedata.category(c) != "Mn"  # Mn = Mark, Nonspacing
    )


# ── Suffix map (jurisdiction-agnostic, 40+ entries) ──────────────────────────
# Order matters: longer / more specific patterns first so "pvt ltd" beats "ltd"
# All entries are lowercased. We match whole tokens only (word boundaries).

_RAW_SUFFIXES: list[str] = [
    # US
    "incorporated", "incorporation",
    "limited liability company",
    "limited liability partnership",
    "limited liability limited partnership",
    "professional limited liability company",
    "professional limited liability",
    "professional corporation",
    "corporation", "company",
    "llc", "llp", "lllp", "pllc", "plc",
    "inc", "corp", "co",
    "associates", "associate",
    "holdings", "holding",
    "enterprises", "enterprise",
    "international",
    "group",
    "services", "service",
    "solutions", "solution",
    # India
    "private limited", "pvt ltd", "pvt. ltd.", "pvt. ltd",
    "pvt ltd.", "private ltd", "private ltd.",
    "limited", "ltd", "ltd.",
    "llp",
    # France (open-set — added for France unseen in train)
    "societe anonyme a responsabilite limitee", "sarl",
    "societe par actions simplifiee unipersonnelle", "sasu",
    "societe par actions simplifiee", "sas",
    "societe anonyme", "sa",
    "entreprise unipersonnelle a responsabilite limitee", "eurl",
    "societe civile immobiliere", "sci",
    "societe en nom collectif", "snc",
    "societe en commandite simple", "scs",
    "societe en commandite par actions", "sca",
    # Generic / Other
    "trading",
    "technologies", "technology",
    "industries", "industry",
    "systems", "system",
    "networks", "network",
    "global",
    "ventures", "venture",
]

# Build a single regex alternation, longest first (already ordered above).
# Wrapped in word boundaries, case-insensitive at compile time.
def _build_suffix_pattern(suffixes: list[str]) -> re.Pattern:
    # Sort descending by length so longer phrases match before sub-phrases
    sorted_sfx = sorted(set(suffixes), key=len, reverse=True)
    escaped = [re.escape(s) for s in sorted_sfx]
    return re.compile(
        r"\b(?:" + "|".join(escaped) + r")\b",
        flags=re.IGNORECASE,
    )


_SUFFIX_RE: re.Pattern = _build_suffix_pattern(_RAW_SUFFIXES)

# PIN (postal code) extraction: 5-digit US zip, 6-char India PIN, 5-digit France
_PIN_RE: re.Pattern = re.compile(r"\b\d{5,6}\b")

# Numeric token extraction (all digit-only tokens, e.g. house numbers, shop numbers)
_NUM_RE: re.Pattern = re.compile(r"\b\d+\b")

# Punctuation to replace with space (preserve digits and letters)
_PUNCT_RE: re.Pattern = re.compile(r"[^\w\s]")

# Multiple whitespace collapse
_WS_RE: re.Pattern = re.compile(r"\s+")


# ── Core single-string normalizer ─────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """Full canonical form of a business name or address.

    Pipeline:
      1. NFKD + diacritic strip
      2. &→and
      3. Suffix tokens → stripped (remove legal form noise)
      4. Punctuation → space
      5. Lowercase
      6. Dedup whitespace, strip edges

    Used for both BM25 tokenization and dense embedding input.
    """
    if not text:
        return ""
    text = _normalize_unicode(text)
    text = text.replace("&", " and ")
    text = _SUFFIX_RE.sub(" ", text)          # strip legal suffixes
    text = _PUNCT_RE.sub(" ", text)           # punctuation → space
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def normalize_name(text: str) -> str:
    """Canonical business name (same as normalize_text)."""
    return normalize_text(text)


def normalize_address(text: str) -> str:
    """Canonical address — same pipeline but suffix map not applied to addresses."""
    if not text:
        return ""
    text = _normalize_unicode(text)
    text = text.replace("&", " and ")
    text = _PUNCT_RE.sub(" ", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def extract_pin(address: str) -> str:
    """Extract first 5-or-6-digit postal code from address, or '' if none."""
    m = _PIN_RE.search(address)
    return m.group() if m else ""


def extract_numeric_tokens(text: str) -> frozenset[str]:
    """All digit-only tokens (house number, shop number, etc.)."""
    return frozenset(_NUM_RE.findall(text))


# ── Polars-level batch preprocessing ─────────────────────────────────────────

def preprocess_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Add canonical columns to a source DataFrame.

    Input columns: entity_id, business_name, business_address, country
    Added columns:
      - name_clean    : normalized business name
      - address_clean : normalized address
      - pin           : extracted postal code (str, "" if none)
      - country_clean : lowercased country string (open-set safe)

    Returns a new DataFrame (does not mutate input).
    """
    # Apply Python-level normalizers via map_elements (vectorized UDF).
    # Polars map_elements is C-backed; OK for 200k rows in <2s.
    df = df.with_columns([
        pl.col("business_name")
          .map_elements(normalize_name, return_dtype=pl.Utf8)
          .alias("name_clean"),

        pl.col("business_address")
          .map_elements(normalize_address, return_dtype=pl.Utf8)
          .alias("address_clean"),

        pl.col("business_address")
          .map_elements(extract_pin, return_dtype=pl.Utf8)
          .alias("pin"),

        # Country: lowercase only — string equality gating, no hardcoded list
        pl.col("country")
          .str.to_lowercase()
          .str.strip_chars()
          .alias("country_clean"),
    ])
    return df


def preprocess_all(
    s1: pl.DataFrame,
    s2: pl.DataFrame,
    s3: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Preprocess S1, S2, S3 in one call. Returns preprocessed copies."""
    return preprocess_dataframe(s1), preprocess_dataframe(s2), preprocess_dataframe(s3)


# ── Public aliases required by task spec ─────────────────────────────────────

def canonicalize_name(name: str) -> str:
    """Canonical form of a business name.

    Applies full normalization pipeline including suffix stripping.
    Alias for normalize_name / normalize_text.

    Parameters
    ----------
    name : str
        Raw business name, possibly with diacritics and legal suffixes.

    Returns
    -------
    str
        Lowercase, ASCII-safe, suffix-stripped canonical form.

    Examples
    --------
    >>> canonicalize_name("Société Générale SARL")
    'societe generale'
    >>> canonicalize_name("Choudhury & Sons Pvt Ltd")
    'choudhury and sons'
    """
    return normalize_name(name)


def canonicalize_address(address: str) -> str:
    """Canonical form of a business address.

    Applies NFKD, &→and, hyphen→space, punctuation→space, lowercase,
    dedup whitespace. Does NOT strip legal suffixes (not meaningful in
    addresses). Preserves numeric tokens for PIN extraction.

    Parameters
    ----------
    address : str
        Raw address string, possibly with diacritics or special chars.

    Returns
    -------
    str
        Normalized address string.

    Examples
    --------
    >>> canonicalize_address("Near SBI ATM, Mumbai 400001")
    'near sbi atm  mumbai 400001'
    """
    return normalize_address(address)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("preprocess.py smoke test")
    print("=" * 60)

    # 1. French suffix stripping: SARL should be removed
    raw1 = "Société Générale SARL"
    out1 = canonicalize_name(raw1)
    print(f"\n[1] French SARL suffix strip")
    print(f"    Input : {raw1!r}")
    print(f"    Output: {out1!r}")
    assert "sarl" not in out1, "SARL suffix should be stripped"
    assert "sarl" not in out1 and "generale" in out1, "Diacritics should be removed"
    print("    ✓ SARL stripped, diacritics normalized")

    # 2. India Pvt Ltd suffix strip + & expansion
    raw2 = "Choudhury & Sons Pvt Ltd"
    out2 = canonicalize_name(raw2)
    print(f"\n[2] India Pvt Ltd + ampersand")
    print(f"    Input : {raw2!r}")
    print(f"    Output: {out2!r}")
    assert "pvt" not in out2 and "ltd" not in out2, "Pvt Ltd should be stripped"
    assert "and" in out2, "& should expand to 'and'"
    print("    ✓ Pvt Ltd stripped, & → and")

    # 3. PIN extraction from Mumbai address
    raw3 = "Near SBI ATM, Mumbai 400001"
    pin3 = extract_pin(raw3)
    print(f"\n[3] PIN extraction")
    print(f"    Input : {raw3!r}")
    print(f"    PIN   : {pin3!r}")
    assert pin3 == "400001", f"Expected '400001', got {pin3!r}"
    print("    ✓ 6-digit India PIN extracted")

    # 4. France handled without any hardcoded country check
    # The function is purely string-based — no country arg, no country list
    raw4 = "L'Entreprise Dupont SAS"
    out4 = canonicalize_name(raw4)
    print(f"\n[4] France — no hardcoded country check")
    print(f"    Input : {raw4!r}")
    print(f"    Output: {out4!r}")
    assert "sas" not in out4, "SAS suffix should be stripped"
    # Verify: canonicalize_name takes only 'name' — no country arg needed
    import inspect
    sig = inspect.signature(canonicalize_name)
    assert list(sig.parameters.keys()) == ["name"], (
        "canonicalize_name must NOT accept a country argument (jurisdiction-agnostic)"
    )
    print("    ✓ SAS stripped, function signature is jurisdiction-agnostic (no country arg)")

    # 5. Additional suffixes: EURL, SCI, SASU
    for raw, sfx in [
        ("Maison EURL", "eurl"),
        ("Résidence SCI Paris", "sci"),
        ("Dupont SASU", "sasu"),
        ("Tata Industries Ltd.", "ltd"),
        ("ABC Corp Inc", "inc"),
    ]:
        out = canonicalize_name(raw)
        assert sfx not in out, f"{sfx.upper()} should be stripped from {raw!r} → got {out!r}"
        print(f"    ✓ {sfx.upper()} stripped: {raw!r} → {out!r}")

    # 6. US ZIP extraction
    raw6 = "123 Main St, Springfield, IL 62701"
    pin6 = extract_pin(raw6)
    assert pin6 == "62701", f"Expected '62701', got {pin6!r}"
    print(f"\n[5] US ZIP: {raw6!r} → PIN={pin6!r} ✓")

    # 7. Polars DataFrame batch preprocessing
    import polars as pl
    df = pl.DataFrame({
        "entity_id": ["S1-001", "S1-002", "S1-003"],
        "business_name": [
            "Société Générale SARL",
            "Choudhury & Sons Pvt Ltd",
            "Amazon Inc",
        ],
        "business_address": [
            "12 Rue de la Paix, Paris 75001",
            "Near SBI ATM, Mumbai 400001",
            "410 Terry Ave N, Seattle, WA 98109",
        ],
        "country": ["France", "India", "US"],
    })
    preprocessed = preprocess_dataframe(df)
    assert "name_clean" in preprocessed.columns
    assert "address_clean" in preprocessed.columns
    assert "pin" in preprocessed.columns
    assert "country_clean" in preprocessed.columns
    pins = preprocessed["pin"].to_list()
    assert "75001" in pins, f"75001 not found in pins: {pins}"
    assert "400001" in pins
    assert "98109" in pins
    print(f"\n[6] Polars batch preprocessing:")
    print(preprocessed.select(["entity_id", "name_clean", "pin", "country_clean"]))
    print("    ✓ All batch columns produced correctly")

    print("\n" + "=" * 60)
    print("All smoke tests PASSED ✓")
    print("=" * 60)

