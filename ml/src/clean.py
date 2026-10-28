"""
Aurelium - Phase 3: Cleaning Pipeline
Turns raw papers.jsonl + enrichment.jsonl into one clean, deduplicated,
labelled and split table (clean_papers.parquet), plus cleaning_log.json
with every count. Raw files are only read, never changed.
"""

import argparse
import html
import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.citation_labels import compute_citation_label, stream_jsonl

Log = Dict[str, Any]

MISSING_POLICY: Dict[str, str] = {
    "id": "drop - without an id a row cannot be joined, deduplicated or traced",
    "publication_year": "drop - needed for the label window and the time split",
    "publication_date": "fill - YYYY-07-01 from publication_year, marked by date_is_imputed",
    "language": "drop - only English is in scope, so unknown language fails the filter",
    "type": "drop - unknown type fails the KEEP_TYPES filter",
    "title": "flag - kept; titles under 3 words are skipped by title dedup",
    "abstract": "flag - has_abstract=False; embeddings fall back to the title",
    "doi": "flag - kept; rows without a DOI are deduplicated by id and title only",
    "is_retracted": "flag - kept as <NA>; the filter drops only an explicit True",
    "is_paratext": "flag - kept as <NA>; the filter drops only an explicit True",
    "venue": "flag - kept as None, venue_canonical is None too",
    "author_ids": "fill - empty list when enrichment is missing; has_enrichment says why",
    "n_authors": "flag - <NA> when enrichment is missing (unknown, not zero)",
    "topic/subfield/field": "flag - kept as None; not used by cleaning",
    "counts_by_year": "fill - empty means 0 citations (same rule as compute_citation_label)",
}

KEEP_TYPES = {"article", "conference-paper", "preprint", "review", "book-chapter"}

SHORT_ABSTRACT_WORDS = 50
BOILERPLATE_PREFIXES = (
    "no abstract",
    "abstract not available",
    "not available",
    "summary form only given",
    "n/a",
)

# Regex (matched on the lowercased venue name) -> canonical name.
# Order matters: the first match wins, so specific patterns come before general ones.
VENUE_ALIASES: Dict[str, str] = {
    r"neural information processing systems|\bnips\b|\bneurips\b": "NeurIPS",
    r"aaai conference on artificial intelligence": "AAAI",
    r"international joint conference on artificial intelligence|\bijcai\b": "IJCAI",
    r"international conference on machine learning$|\bicml\b": "ICML",
    r"proceedings of machine learning research": "PMLR",
    r"international conference on learning representations|\biclr\b": "ICLR",
    r"computer vision and pattern recognition workshops?|\bcvprw\b": "CVPR Workshops",
    r"computer vision and pattern recognition|\bcvpr\b": "CVPR",
    r"international conference on computer vision$|\biccv\b": "ICCV",
    r"findings of the association for computational linguistics": "Findings of ACL",
    r"empirical methods in natural language processing|\bemnlp\b": "EMNLP",
    r"north american chapter of the association for computational linguistics|\bnaacl\b": "NAACL",
    r"transactions of the association for computational linguistics": "TACL",
    r"annual meeting of the association for computational linguistics|^acl\b": "ACL",
}

MIN_TITLE_WORDS_FOR_DEDUP = 3

# Same years as src/citation_model.py so all phases agree.
SPLIT_YEARS: Dict[str, range] = {
    "train": range(2015, 2021),
    "val": range(2021, 2022),
    "test": range(2022, 2024),
}

COLUMN_ORDER: List[str] = [
    "id", "split", "publication_year", "publication_date", "date_is_imputed", "date_year_mismatch",
    "type", "language", "title", "abstract", "title_raw", "abstract_raw",
    "has_abstract", "abstract_word_count", "abstract_short", "abstract_boilerplate",
    "doi", "doi_norm", "venue", "venue_canonical", "venue_id", "venue_type",
    "author_ids", "n_authors", "is_retracted", "is_paratext", "has_enrichment",
    "topic", "subfield", "field", "referenced_works",
    "citations_3yr", "log_citations_3yr", "label_usable",
    "pre_pub_citations", "impossible_citations", "window_truncated",
]

INLINE_TAG = re.compile(r"</?(?:jats:)?(?:i|b|em|strong|sub|sup|italic|bold|scp|u)\b[^<>]*>", re.I)
ANY_TAG = re.compile(r"</?[a-zA-Z][\w:-]*(?:\s[^<>]*)?/?>")
LATEX_CMD = re.compile(r"\\(?:emph|textbf|textit|textrm|mathrm|mathbf|mathit|text)\{([^{}]*)\}")
LATEX_MATH = re.compile(r"\$(\S(?:[^$]*\S)?)\$")


# ---------- 1. load_and_join ----------

def load_and_join(raw_dir: str, log: Log) -> pd.DataFrame:
    """Load papers and enrichment, left-join on id, and report papers with no enrichment."""
    papers_path = os.path.join(raw_dir, "papers.jsonl")
    enrich_path = os.path.join(raw_dir, "enrichment.jsonl")
    if not os.path.exists(enrich_path):
        raise FileNotFoundError(f"{enrich_path} not found. Run python -m src.enrich first.")

    papers = pd.DataFrame(list(stream_jsonl(papers_path)))
    no_id = papers["id"].isna()
    papers = papers[~no_id]
    enrich = pd.DataFrame(list(stream_jsonl(enrich_path))).drop_duplicates("id")

    df = papers.merge(enrich, on="id", how="left", indicator=True)
    df["has_enrichment"] = df.pop("_merge") == "both"
    df["author_ids"] = df["author_ids"].map(lambda v: v if isinstance(v, list) else [])
    df["n_authors"] = df["n_authors"].astype("Int64")
    df["is_retracted"] = df["is_retracted"].astype("boolean")
    df["is_paratext"] = df["is_paratext"].astype("boolean")

    missing = df.loc[~df["has_enrichment"], "id"].tolist()
    log["load_and_join"] = {
        "papers_rows": len(papers) + int(no_id.sum()),
        "dropped_no_id": int(no_id.sum()),
        "enrichment_rows": len(enrich),
        "papers_without_enrichment": len(missing),
        "papers_without_enrichment_ids": missing,
    }
    print(f"[1] load_and_join: {len(df)} papers | without enrichment: {len(missing)}")
    return df.reset_index(drop=True)


# ---------- 2. fix_types ----------

def fix_types(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    """Year to int, date to datetime (YYYY-07-01 if only the year exists), flag year/date mismatch."""
    year = pd.to_numeric(df["publication_year"], errors="coerce")
    no_year = year.isna()
    df = df[~no_year].copy()
    df["publication_year"] = year[~no_year].astype(int)

    date = pd.to_datetime(df["publication_date"], format="%Y-%m-%d", errors="coerce")
    df["date_is_imputed"] = date.isna()
    fallback = pd.to_datetime(df["publication_year"].astype(str) + "-07-01")
    df["publication_date"] = date.fillna(fallback)
    df["date_year_mismatch"] = ~df["date_is_imputed"] & (df["publication_date"].dt.year != df["publication_year"])

    log["fix_types"] = {
        "dropped_no_year": int(no_year.sum()),
        "dates_imputed": int(df["date_is_imputed"].sum()),
        "date_year_mismatch": int(df["date_year_mismatch"].sum()),
    }
    print(f"[2] fix_types: removed {int(no_year.sum())} (no year) | "
          f"dates imputed {log['fix_types']['dates_imputed']} | mismatch {log['fix_types']['date_year_mismatch']}")
    return df


# ---------- 3. filter_scope ----------

def filter_scope(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    """Keep English, non-retracted, non-paratext papers whose type is in KEEP_TYPES."""
    type_counts = df["type"].value_counts(dropna=False)
    print("[3] filter_scope: type distribution before filtering")
    print(type_counts.to_string())

    dropped_types = df.loc[~df["type"].isin(KEEP_TYPES), "type"].value_counts(dropna=False)
    other = df[df["type"] == "other"]
    other_rescue = other["doi"].notna() & other["venue_type"].isin(["journal", "conference"])

    rules = [
        ("language_not_en", lambda d: d["language"] != "en"),
        ("retracted", lambda d: d["is_retracted"].eq(True).fillna(False)),
        ("paratext", lambda d: d["is_paratext"].eq(True).fillna(False)),
        ("type_not_kept", lambda d: ~d["type"].isin(KEEP_TYPES)),
    ]
    removed: Dict[str, int] = {}
    for name, rule in rules:
        mask = rule(df)
        removed[name] = int(mask.sum())
        df = df[~mask]
        print(f"    removed {removed[name]:5d} | {name}")

    log["filter_scope"] = {
        "type_distribution": {str(k): int(v) for k, v in type_counts.items()},
        "keep_types": sorted(KEEP_TYPES),
        "removed": removed,
        "dropped_by_type": {str(k): int(v) for k, v in dropped_types.items()},
        "other_with_doi_and_journal_or_conference": int(other_rescue.sum()),
        "rows_after": len(df),
    }
    print(f"    'other' rows that look like real papers (DOI + journal/conference): {int(other_rescue.sum())}")
    return df


# ---------- 4. clean_text ----------

def clean_string(text: Any) -> Optional[str]:
    """Unescape HTML, strip tags and simple LaTeX, NFKC-normalize and collapse whitespace."""
    if not isinstance(text, str):
        return None
    text = html.unescape(text)
    text = INLINE_TAG.sub("", text)  # <sub>2</sub> -> 2, so H<sub>2</sub>O -> H2O
    text = ANY_TAG.sub(" ", text)  # block tags like <jats:p> separate words
    text = LATEX_CMD.sub(r"\1", text)
    text = LATEX_MATH.sub(r"\1", text)
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text or None


def clean_text(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    """Clean title and abstract, keeping the originals in title_raw and abstract_raw."""
    df = df.copy()
    df["title_raw"] = df["title"]
    df["abstract_raw"] = df["abstract"]
    df["title"] = df["title_raw"].map(clean_string)
    df["abstract"] = df["abstract_raw"].map(clean_string)

    changed = {
        col: int((df[col].fillna("") != df[f"{col}_raw"].fillna("")).sum())
        for col in ("title", "abstract")
    }
    log["clean_text"] = {"titles_changed": changed["title"], "abstracts_changed": changed["abstract"]}
    print(f"[4] clean_text: titles changed {changed['title']} | abstracts changed {changed['abstract']}")
    return df


# ---------- 5. abstract_quality ----------

def abstract_quality(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    """Flag missing, short and boilerplate abstracts. Nothing is dropped."""
    df = df.copy()
    abstract_lower = df["abstract"].fillna("").str.lower()
    df["has_abstract"] = df["abstract"].notna()
    df["abstract_word_count"] = df["abstract"].fillna("").str.split().str.len()
    df["abstract_short"] = df["has_abstract"] & (df["abstract_word_count"] < SHORT_ABSTRACT_WORDS)
    same_as_title = df["has_abstract"] & (abstract_lower == df["title"].fillna("").str.lower())
    df["abstract_boilerplate"] = same_as_title | abstract_lower.str.startswith(BOILERPLATE_PREFIXES)

    log["abstract_quality"] = {
        "missing": int((~df["has_abstract"]).sum()),
        "short": int(df["abstract_short"].sum()),
        "boilerplate": int(df["abstract_boilerplate"].sum()),
    }
    q = log["abstract_quality"]
    print(f"[5] abstract_quality: removed 0 | missing {q['missing']} | short {q['short']} | boilerplate {q['boilerplate']}")
    return df


# ---------- 6. normalize_venue ----------

def canonical_venue(name: Any) -> Optional[str]:
    """Map a venue name to its canonical form via VENUE_ALIASES, else return it unchanged."""
    if not isinstance(name, str):
        return None
    lower = name.lower().strip()
    for pattern, canonical in VENUE_ALIASES.items():
        if re.search(pattern, lower):
            return canonical
    return name


def normalize_venue(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    """Add venue_canonical and print the 40 most common raw venue names."""
    df = df.copy()
    df["venue_canonical"] = df["venue"].map(canonical_venue)
    changed = df["venue"].notna() & (df["venue_canonical"] != df["venue"])

    top = df["venue"].value_counts().head(40)
    log["normalize_venue"] = {
        "venues_changed": int(changed.sum()),
        "canonical_counts": {str(k): int(v) for k, v in df.loc[changed, "venue_canonical"].value_counts().items()},
        "top40_raw_venues": {str(k): int(v) for k, v in top.items()},
    }
    print(f"[6] normalize_venue: changed {int(changed.sum())} | top 40 raw venues:")
    print(top.to_string())
    return df


# ---------- 7. deduplicate ----------

def normalize_doi(doi: Any) -> Optional[str]:
    """Lowercase and strip the https://doi.org/ prefix, so both copies of a DOI compare equal."""
    if not isinstance(doi, str):
        return None
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip().lower())
    return doi or None


def normalize_title(title: Any, year: int) -> Optional[str]:
    """Lowercase, drop punctuation, add the year. Titles under 3 words return None (too generic)."""
    if not isinstance(title, str):
        return None
    words = re.sub(r"[^\w\s]", " ", title.lower()).split()
    if len(words) < MIN_TITLE_WORDS_FOR_DEDUP:
        return None
    return " ".join(words) + f"|{year}"


def drop_duplicates_by(df: pd.DataFrame, key: str, rule: str, log: Log) -> pd.DataFrame:
    """
    Within each group sharing `key`, keep the row with an abstract, then the earliest date.
    Rows where `key` is missing are never merged. Every dropped id -> kept id is logged.
    """
    ordered = df.sort_values(["has_abstract", "publication_date", "id"], ascending=[False, True, True])
    keyed = ordered[ordered[key].notna()]
    is_dup = keyed.duplicated(key, keep="first")
    kept_id = keyed.groupby(key)["id"].transform("first")

    pairs = [{"dropped": d, "kept": k, "rule": rule} for d, k in zip(keyed.loc[is_dup, "id"], kept_id[is_dup])]
    log["deduplicate"]["pairs"].extend(pairs)
    log["deduplicate"]["removed"][rule] = len(pairs)
    print(f"    removed {len(pairs):5d} | by {rule}")
    return df.drop(index=keyed.index[is_dup])


def deduplicate(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    """Deduplicate by id, then normalized DOI, then normalized title + year."""
    df = df.copy()
    df["doi_norm"] = df["doi"].map(normalize_doi)
    df["title_norm"] = [normalize_title(t, y) for t, y in zip(df["title"], df["publication_year"])]

    log["deduplicate"] = {"removed": {}, "pairs": []}
    print("[7] deduplicate:")
    for key, rule in [("id", "id"), ("doi_norm", "doi"), ("title_norm", "title_year")]:
        df = drop_duplicates_by(df, key, rule, log)
    return df.drop(columns="title_norm")


# ---------- 8. citation_label ----------

def citations_before(counts_by_year: Any, year: int) -> int:
    """Sum of citations dated strictly before `year`."""
    if not isinstance(counts_by_year, list):
        return 0
    return sum(e.get("cited_by_count", 0) for e in counts_by_year if e.get("year") is not None and e["year"] < year)


def earliest_citation_year(df: pd.DataFrame) -> Optional[int]:
    """Earliest year present in any counts_by_year: OpenAlex reports nothing before it."""
    years = [e["year"] for cby in df["counts_by_year"] if isinstance(cby, list) for e in cby if e.get("year")]
    return min(years) if years else None


def citation_label(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    """Add the 3-year label via compute_citation_label, plus flags for known label problems."""
    df = df.copy()
    cby = [c if isinstance(c, list) else None for c in df["counts_by_year"]]
    years = df["publication_year"].tolist()

    raw_label = pd.Series([compute_citation_label(c, y) for c, y in zip(cby, years)], index=df.index, dtype="Int64")
    df["pre_pub_citations"] = [citations_before(c, y) for c, y in zip(cby, years)]
    df["impossible_citations"] = [citations_before(c, y - 1) > 0 for c, y in zip(cby, years)]

    earliest = earliest_citation_year(df)
    df["window_truncated"] = df["publication_year"] < earliest if earliest is not None else False
    df["label_usable"] = ~df["window_truncated"]

    # Truncated windows are undercounted, so their label is set to <NA> rather than kept silently.
    df["citations_3yr"] = raw_label.where(df["label_usable"])
    df["log_citations_3yr"] = np.log1p(df["citations_3yr"].to_numpy(dtype=float, na_value=np.nan))

    truncated = df.loc[df["window_truncated"], "publication_year"].value_counts().sort_index()
    log["citation_label"] = {
        "earliest_citation_year_in_data": earliest,
        "rows_with_pre_pub_citations": int((df["pre_pub_citations"] > 0).sum()),
        "pre_pub_citations_total": int(df["pre_pub_citations"].sum()),
        "rows_with_impossible_citations": int(df["impossible_citations"].sum()),
        "window_truncated_rows": int(df["window_truncated"].sum()),
        "window_truncated_by_cohort": {int(k): int(v) for k, v in truncated.items()},
    }
    c = log["citation_label"]
    print(f"[8] citation_label: removed 0 | earliest citation year in data: {earliest} | "
          f"pre-pub rows {c['rows_with_pre_pub_citations']} | impossible rows {c['rows_with_impossible_citations']}")
    if c["window_truncated_rows"]:
        print(f"    WARNING: {c['window_truncated_rows']} labels truncated and set to <NA>, by cohort: "
              f"{c['window_truncated_by_cohort']}")
    else:
        print("    no cohort has a truncated citation window")
    return df


# ---------- 9. split ----------

def year_to_split(year: int) -> str:
    """Map a publication year to train / val / test (or out_of_range)."""
    for name, years in SPLIT_YEARS.items():
        if year in years:
            return name
    return "out_of_range"


def split(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    """Add the time-based split column, using the same years as citation_model.py."""
    df = df.copy()
    df["split"] = df["publication_year"].map(year_to_split)
    counts = df["split"].value_counts()
    log["split"] = {str(k): int(v) for k, v in counts.items()}
    print(f"[9] split: {log['split']}")
    return df


# ---------- 10. save ----------

def save(df: pd.DataFrame, log: Log, out_dir: str) -> None:
    """Write clean_papers.parquet with a fixed column order and cleaning_log.json."""
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, "clean_papers.parquet")
    log_path = os.path.join(out_dir, "cleaning_log.json")

    df[COLUMN_ORDER].reset_index(drop=True).to_parquet(parquet_path, index=False)
    log["save"] = {"rows": len(df), "columns": COLUMN_ORDER}
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"[10] save: {len(df)} rows -> {parquet_path} | log -> {log_path}")


def run(raw_dir: str, out_dir: str) -> pd.DataFrame:
    """Apply all cleaning rules in order and save the result."""
    log: Log = {"raw_dir": raw_dir, "missing_policy": MISSING_POLICY}
    df = load_and_join(raw_dir, log)
    for rule in (fix_types, filter_scope, clean_text, abstract_quality,
                 normalize_venue, deduplicate, citation_label, split):
        df = rule(df, log)
    save(df, log, out_dir)
    return df


def main() -> None:
    """CLI entry point for the cleaning pipeline."""
    parser = argparse.ArgumentParser(description="Aurelium Cleaning Pipeline")
    parser.add_argument("--raw-dir", default="data/raw", help="Folder with papers.jsonl and enrichment.jsonl")
    parser.add_argument("--out-dir", default="data/clean", help="Folder for clean_papers.parquet and cleaning_log.json")
    parser.add_argument("--test", action="store_true", help="Clean data/test/raw into data/test/clean")
    args = parser.parse_args()

    if args.test:
        args.raw_dir, args.out_dir = "data/test/raw", "data/test/clean"

    print(f"=== Aurelium Cleaning {'(TEST MODE)' if args.test else ''} ===")
    print("Missing-value policy:")
    for column, policy in MISSING_POLICY.items():
        print(f"  {column:22s} {policy}")
    print()
    run(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
