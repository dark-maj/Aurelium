"""
Aurelium - Phase 3: Data-Quality Report
Reads clean_papers.parquet and cleaning_log.json and writes a Markdown report:
cleaning funnel, per-year table, top types/venues/topics and known data issues.
"""

import argparse
import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

Log = Dict[str, Any]
TOP_N = 15
N_EXAMPLES = 3


def md_table(headers: List[str], rows: List[List[Any]]) -> List[str]:
    """Render a Markdown table as a list of lines."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def pct(part: float, whole: float) -> str:
    """Percentage with one decimal, or n/a when the whole is zero."""
    return f"{part / whole * 100:.1f}%" if whole else "n/a"


def funnel_steps(log: Log) -> List[Tuple[str, int]]:
    """(rule, rows removed) in pipeline order, read from cleaning_log.json."""
    scope = log["filter_scope"]["removed"]
    dedup = log["deduplicate"]["removed"]
    return [
        ("1. load_and_join: no id", log["load_and_join"]["dropped_no_id"]),
        ("2. fix_types: no publication_year", log["fix_types"]["dropped_no_year"]),
        ("3. filter_scope: language not en", scope["language_not_en"]),
        ("3. filter_scope: retracted", scope["retracted"]),
        ("3. filter_scope: paratext", scope["paratext"]),
        ("3. filter_scope: type not in KEEP_TYPES", scope["type_not_kept"]),
        ("4. clean_text (changes text only)", 0),
        ("5. abstract_quality (flags only)", 0),
        ("6. normalize_venue (adds a column only)", 0),
        ("7. deduplicate: same id", dedup["id"]),
        ("7. deduplicate: same DOI", dedup["doi"]),
        ("7. deduplicate: same title + year", dedup["title_year"]),
        ("8. citation_label (flags only)", 0),
        ("9. split (adds a column only)", 0),
    ]


def funnel_section(log: Log) -> List[str]:
    """Raw rows -> rows left after each rule -> final, with percentages of raw."""
    raw = log["load_and_join"]["papers_rows"]
    remaining = raw
    rows: List[List[Any]] = [["raw papers.jsonl", "", f"{raw:,}", "100.0%"]]
    for step, removed in funnel_steps(log):
        remaining -= removed
        rows.append([step, f"-{removed:,}" if removed else "0", f"{remaining:,}", pct(remaining, raw)])

    final = log["save"]["rows"]
    rows.append(["**final clean_papers.parquet**", "", f"**{final:,}**", f"**{pct(final, raw)}**"])
    lines = ["## 1. Cleaning funnel", ""] + md_table(["Step", "Removed", "Rows left", "% of raw"], rows)
    if remaining != final:
        lines += ["", f"> WARNING: funnel arithmetic gives {remaining:,} but the parquet has {final:,} rows."]

    lines += ["", f"Papers with no enrichment row (kept, `has_enrichment=False`): "
                  f"{log['load_and_join']['papers_without_enrichment']:,}  "]
    lines += [f"`other` rows dropped that had a DOI and a journal/conference venue: "
              f"{log['filter_scope']['other_with_doi_and_journal_or_conference']:,}  "]
    lines += [f"Text changed by clean_text: {log['clean_text']['titles_changed']:,} titles, "
              f"{log['clean_text']['abstracts_changed']:,} abstracts"]
    return lines


def per_year_section(df: pd.DataFrame) -> List[str]:
    """Papers, abstract coverage, truncation and median label per publication year."""
    rows = []
    for year, g in df.groupby("publication_year"):
        median = g["citations_3yr"].dropna().median()
        rows.append([
            year, f"{len(g):,}",
            pct(g["has_abstract"].sum(), len(g)),
            pct(g["abstract_short"].sum(), len(g)),
            pct(g["window_truncated"].sum(), len(g)),
            "n/a" if pd.isna(median) else f"{median:g}",
            ", ".join(sorted(g["split"].unique())),
        ])
    headers = ["Year", "Papers", "% abstract", "% short abstract", "% window truncated", "Median citations_3yr", "Split"]
    return ["## 2. Per publication year", ""] + md_table(headers, rows)


def top_section(df: pd.DataFrame) -> List[str]:
    """Top types, canonical venues and topics after cleaning."""
    lines = ["## 3. Most common values after cleaning", ""]
    for column, label in [("type", "Type"), ("venue_canonical", "Venue (canonical)"), ("topic", "Topic")]:
        counts = df[column].value_counts().head(TOP_N)
        rows = [[name, f"{n:,}", pct(n, len(df))] for name, n in counts.items()]
        missing = int(df[column].isna().sum())
        lines += [f"### Top {TOP_N} by {column}", ""] + md_table([label, "Papers", "%"], rows)
        lines += ["", f"Missing: {missing:,} ({pct(missing, len(df))})", ""]
    return lines


def examples(ids: pd.Series) -> str:
    """First N example ids, as short W-ids, or 'none'."""
    short = [i.rsplit("/", 1)[-1] for i in ids.head(N_EXAMPLES)]
    return ", ".join(f"`{s}`" for s in short) if short else "none"


def issue_window_reaches(df: pd.DataFrame, latest: Optional[int]) -> Tuple[pd.Series, str]:
    """Papers whose label window (year .. year+2) reaches the partially counted latest year."""
    if latest is None:
        return df["id"].iloc[0:0], "unknown: rerun `python -m src.clean` to log latest_citation_year_in_data"
    reaches = df[df["publication_year"] + 2 >= latest]["id"]
    status = "confirmed: no label window reaches it" if reaches.empty else "PROBLEM: these labels include a partial year"
    return reaches, status


def issues_section(df: pd.DataFrame, log: Log) -> List[str]:
    """Known data issues with counts and example ids."""
    labels = log["citation_label"]
    latest = labels.get("latest_citation_year_in_data")
    partial, partial_status = issue_window_reaches(df, latest)
    truncated = df[df["window_truncated"]]

    rows = [
        ["Pre-publication citations (preprint merged with journal version)",
         f"{(df['pre_pub_citations'] > 0).sum():,} papers, {int(df['pre_pub_citations'].sum()):,} citations",
         "Ignored by the label (window starts at publication_year)",
         examples(df.loc[df["pre_pub_citations"] > 0, "id"])],
        ["Impossible citations (> 1 year before publication)",
         f"{int(df['impossible_citations'].sum()):,} papers",
         "Flag only; likely a wrong publication year or a merged record",
         examples(df.loc[df["impossible_citations"], "id"])],
        ["Truncated label window (starts before earliest year in counts_by_year)",
         f"{len(truncated):,} papers; earliest year {labels['earliest_citation_year_in_data']}; "
         f"by cohort {labels['window_truncated_by_cohort'] or 'none'}",
         "Label set to <NA>, label_usable=False",
         examples(truncated["id"])],
        [f"Partial current year ({latest} in counts_by_year is incomplete)",
         f"{len(partial):,} papers; latest window ends {int(df['publication_year'].max()) + 2}",
         partial_status,
         examples(partial)],
    ]
    return ["## 4. Known data issues", ""] + md_table(["Issue", "Count", "Handling", "Examples"], rows)


def build_report(df: pd.DataFrame, log: Log, clean_dir: str) -> str:
    """Assemble all report sections into one Markdown string."""
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Aurelium - Data-Quality Report (Phase 3)",
        "",
        f"Generated {now} by `python -m src.data_quality` from `{clean_dir}` ({len(df):,} papers).",
        "",
    ]
    for section in (funnel_section(log), per_year_section(df), top_section(df), issues_section(df, log)):
        lines += section + [""]
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for the data-quality report."""
    parser = argparse.ArgumentParser(description="Aurelium Data-Quality Report")
    parser.add_argument("--clean-dir", default="data/clean", help="Folder with clean_papers.parquet and cleaning_log.json")
    parser.add_argument("--out", default="docs/data_quality.md", help="Markdown report path")
    parser.add_argument("--test", action="store_true", help="Report on data/test/clean (writes to data/test/reports)")
    args = parser.parse_args()

    if args.test:
        args.clean_dir, args.out = "data/test/clean", "data/test/reports/data_quality.md"

    df = pd.read_parquet(os.path.join(args.clean_dir, "clean_papers.parquet"))
    with open(os.path.join(args.clean_dir, "cleaning_log.json"), encoding="utf-8") as f:
        log = json.load(f)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_report(df, log, args.clean_dir))
    print(f"Wrote {args.out} ({len(df)} papers)")


if __name__ == "__main__":
    main()
