"""
Aurelium - Phase 3: tests for src.clean
Tiny hand-made rows with known answers. No network calls.
"""

import json
from typing import Any, Dict, List

import pandas as pd

from src.clean import (
    abstract_quality,
    canonical_venue,
    citation_label,
    clean_string,
    deduplicate,
    fix_types,
    load_and_join,
    normalize_doi,
    split,
)


def new_log() -> Dict[str, Any]:
    """Empty log dict, as run() would pass to each rule."""
    return {}


# ---------- clean_text ----------

def test_clean_string_html_and_jats() -> None:
    """Entities are unescaped, inline tags vanish, block tags become spaces."""
    assert clean_string("H<sub>2</sub>O &amp; <i>deep</i> nets") == "H2O & deep nets"
    assert clean_string("<jats:p>First.</jats:p><jats:p>Second.</jats:p>") == "First. Second."
    assert clean_string("&lt;i&gt;Nested&lt;/i&gt; escape") == "Nested escape"


def test_clean_string_latex_unicode_whitespace() -> None:
    """Simple LaTeX is unwrapped, NFKC fixes ligatures, whitespace collapses, casing is kept."""
    assert clean_string(r"We study \emph{Robust} $O(n)$ \textbf{bounds}") == "We study Robust O(n) bounds"
    assert clean_string("ﬁne-tuning   with\np < 0.05") == "fine-tuning with p < 0.05"


def test_clean_string_keeps_prices_and_handles_missing() -> None:
    """Dollar amounts are not maths; None and blank text become None."""
    assert clean_string("costs $5 and $10") == "costs $5 and $10"
    assert clean_string(None) is None
    assert clean_string("   ") is None


# ---------- fix_types ----------

def test_year_only_date_is_imputed() -> None:
    """A string year with no date becomes int and gets YYYY-07-01, flagged as imputed."""
    df = pd.DataFrame({"publication_year": ["2020", 2019], "publication_date": [None, "2019-03-04"]})
    out = fix_types(df, new_log())
    assert out["publication_year"].tolist() == [2020, 2019]
    assert out["publication_date"].iloc[0] == pd.Timestamp("2020-07-01")
    assert out["date_is_imputed"].tolist() == [True, False]
    assert not out["date_year_mismatch"].any()


def test_missing_year_is_dropped() -> None:
    """A row without a usable year cannot be split or labelled."""
    df = pd.DataFrame({"publication_year": [None, "abc", 2020], "publication_date": [None, None, None]})
    log = new_log()
    out = fix_types(df, log)
    assert len(out) == 1
    assert log["fix_types"]["dropped_no_year"] == 2


# ---------- abstract_quality ----------

def test_abstract_flags() -> None:
    """Short, missing and boilerplate abstracts are flagged, never dropped."""
    df = pd.DataFrame({
        "title": ["A title", "Same text", "Keynote"],
        "abstract": ["word " * 10, "Same text", "Summary form only given, as follows."],
    })
    df.loc[3] = ["No abstract here", None]
    out = abstract_quality(df, new_log())
    assert len(out) == 4
    assert out["has_abstract"].tolist() == [True, True, True, False]
    assert out["abstract_short"].tolist() == [True, True, True, False]
    assert out["abstract_boilerplate"].tolist() == [False, True, True, False]


# ---------- normalize_venue ----------

def test_venue_aliases() -> None:
    """Known variants map to one name; lookalikes and unknown venues are left alone."""
    assert canonical_venue("NIPS") == "NeurIPS"
    assert canonical_venue("Advances in Neural Information Processing Systems") == "NeurIPS"
    assert canonical_venue("Proceedings of the AAAI Conference on Artificial Intelligence") == "AAAI"
    assert canonical_venue("Findings of the Association for Computational Linguistics: EMNLP 2020") == "Findings of ACL"
    assert canonical_venue("International Conference on Machine Learning and Applications") == (
        "International Conference on Machine Learning and Applications"
    )
    assert canonical_venue(None) is None


# ---------- deduplicate ----------

def dedup_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Build rows with the columns deduplicate() needs."""
    df = pd.DataFrame(rows)
    df["has_abstract"] = df["abstract"].notna()
    df["publication_date"] = pd.to_datetime(df["publication_date"])
    return df


def test_doi_dedup_keeps_copy_with_abstract() -> None:
    """Same DOI written two ways: the copy with an abstract wins, even though it is later."""
    df = dedup_rows([
        {"id": "W1", "doi": "https://doi.org/10.1/ABC", "title": "Early copy without text",
         "abstract": None, "publication_year": 2019, "publication_date": "2019-01-01"},
        {"id": "W2", "doi": "10.1/abc", "title": "Later copy with text",
         "abstract": "Real abstract.", "publication_year": 2019, "publication_date": "2019-06-01"},
    ])
    log = new_log()
    out = deduplicate(df, log)
    assert out["id"].tolist() == ["W2"]
    assert log["deduplicate"]["pairs"] == [{"dropped": "W1", "kept": "W2", "rule": "doi"}]


def test_title_dedup_ignores_case_punctuation_but_not_short_titles() -> None:
    """Same title + year merges; two different 'Editorial' rows are not merged."""
    df = dedup_rows([
        {"id": "W1", "doi": None, "title": "Deep Learning: A Survey", "abstract": "x",
         "publication_year": 2020, "publication_date": "2020-01-01"},
        {"id": "W2", "doi": None, "title": "deep learning - a survey!", "abstract": "y",
         "publication_year": 2020, "publication_date": "2020-05-01"},
        {"id": "W3", "doi": None, "title": "Editorial", "abstract": None,
         "publication_year": 2020, "publication_date": "2020-01-01"},
        {"id": "W4", "doi": None, "title": "Editorial", "abstract": None,
         "publication_year": 2020, "publication_date": "2020-02-01"},
    ])
    out = deduplicate(df, new_log())
    assert sorted(out["id"]) == ["W1", "W3", "W4"]


def test_normalize_doi() -> None:
    """Prefix and case differences disappear."""
    assert normalize_doi("https://dx.doi.org/10.1145/ABC") == "10.1145/abc"
    assert normalize_doi(None) is None


# ---------- citation_label ----------

def test_pre_publication_citations_are_counted_not_labelled() -> None:
    """Citations before the publication year are flagged and excluded from the label."""
    df = pd.DataFrame({
        "publication_year": [2020],
        "counts_by_year": [[
            {"year": 2017, "cited_by_count": 1},
            {"year": 2019, "cited_by_count": 5},
            {"year": 2020, "cited_by_count": 2},
            {"year": 2022, "cited_by_count": 3},
            {"year": 2023, "cited_by_count": 9},
        ]],
    })
    out = citation_label(df, new_log())
    assert out["citations_3yr"].iloc[0] == 5  # 2020 + 2022; 2023 is outside the window
    assert out["pre_pub_citations"].iloc[0] == 6  # 2017 + 2019
    assert bool(out["impossible_citations"].iloc[0])  # 2017 is more than 1 year early


def test_truncated_window_label_is_missing() -> None:
    """If OpenAlex starts at 2017, a 2015 paper's label is unusable; a 2017 paper's is fine."""
    df = pd.DataFrame({
        "publication_year": [2015, 2017],
        "counts_by_year": [[{"year": 2017, "cited_by_count": 4}], [{"year": 2018, "cited_by_count": 1}]],
    })
    log = new_log()
    out = citation_label(df, log)
    assert out["window_truncated"].tolist() == [True, False]
    assert out["label_usable"].tolist() == [False, True]
    assert pd.isna(out["citations_3yr"].iloc[0])
    assert out["citations_3yr"].iloc[1] == 1
    assert log["citation_label"]["window_truncated_by_cohort"] == {2015: 1}


# ---------- split ----------

def test_split_years_match_citation_model() -> None:
    """Train 2015-2020, val 2021, test 2022-2023."""
    df = pd.DataFrame({"publication_year": [2015, 2020, 2021, 2022, 2023, 2024]})
    out = split(df, new_log())
    assert out["split"].tolist() == ["train", "train", "val", "test", "test", "out_of_range"]


# ---------- load_and_join ----------

def test_left_join_reports_missing_enrichment(tmp_path) -> None:
    """A paper without an enrichment row is kept and reported."""
    papers = [{"id": "W1", "title": "a"}, {"id": "W2", "title": "b"}]
    enrichment = [{"id": "W1", "doi": None, "is_retracted": False, "is_paratext": False, "venue": None,
                   "venue_id": None, "venue_type": None, "author_ids": ["A1"], "n_authors": 1}]
    for name, rows in [("papers.jsonl", papers), ("enrichment.jsonl", enrichment)]:
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    log = new_log()
    out = load_and_join(str(tmp_path), log)
    assert out["id"].tolist() == ["W1", "W2"]
    assert out["has_enrichment"].tolist() == [True, False]
    assert out["author_ids"].tolist() == [["A1"], []]
    assert log["load_and_join"]["papers_without_enrichment_ids"] == ["W2"]
