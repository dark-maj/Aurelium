"""
Aurelium - Phase 2: Citation Label Extraction & Coverage Validation
Builds the 3-year citation impact target (y and log1p(y)) from OpenAlex counts_by_year.
Strictly isolates citation counts to prevent label leakage into feature stores.
"""

import collections
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def compute_citation_label(
    counts_by_year: Optional[List[Dict[str, Any]]],
    pub_year: Optional[int],
) -> int:
    """
    Compute total citations in the publication year plus the next 2 years:
    y = sum_{t in [pub_year, pub_year+1, pub_year+2]} cited_by_count(t).

    Rules:
    - Ignore citation entries dated before pub_year (the arXiv preprint effect).
    - Ignore citation entries dated after pub_year + 2 (ensures fair cohort comparison).
    - If counts_by_year is empty or None, citation count is 0.
    """
    if not counts_by_year or pub_year is None:
        return 0

    window = {pub_year, pub_year + 1, pub_year + 2}
    total_citations = 0

    for entry in counts_by_year:
        year = entry.get("year")
        count = entry.get("cited_by_count", 0)
        if year is not None and year in window:
            total_citations += count

    return int(total_citations)


def build_labels(
    records: Iterable[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a DataFrame of paper IDs, publication years, raw 3-year citations,
    and log-transformed targets log1p(y).

    Memory-safe: iterates record generator without loading full JSON objects into memory.
    """
    rows = []
    for r in records:
        pid = r.get("id")
        pub_year = r.get("publication_year")
        if not pid or pub_year is None:
            continue

        raw_y = compute_citation_label(r.get("counts_by_year"), pub_year)
        rows.append({
            "id": pid,
            "publication_year": int(pub_year),
            "citations_3yr": raw_y,
            "log_citations_3yr": float(np.log1p(raw_y)),
        })

    df = pd.DataFrame(rows)
    return df


def check_coverage(
    records: Iterable[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    """
    Measure counts_by_year coverage per publication year cohort.
    Checks whether older years (2015, 2016) or recent years have complete
    observation windows in counts_by_year.
    """
    cohort_stats: Dict[int, Dict[str, Any]] = collections.defaultdict(
        lambda: {
            "total_papers": 0,
            "has_cby_entries": 0,
            "has_window_citations": 0,
            "min_cby_year": 9999,
            "max_cby_year": -9999,
        }
    )

    for r in records:
        pub_year = r.get("publication_year")
        if pub_year is None:
            continue
        pub_year = int(pub_year)
        stats = cohort_stats[pub_year]
        stats["total_papers"] += 1

        cby = r.get("counts_by_year") or []
        if cby:
            stats["has_cby_entries"] += 1
            years = [e["year"] for e in cby if "year" in e]
            if years:
                stats["min_cby_year"] = min(stats["min_cby_year"], min(years))
                stats["max_cby_year"] = max(stats["max_cby_year"], max(years))

            window = {pub_year, pub_year + 1, pub_year + 2}
            window_sum = sum(e.get("cited_by_count", 0) for e in cby if e.get("year") in window)
            if window_sum > 0:
                stats["has_window_citations"] += 1

    report: Dict[int, Dict[str, Any]] = {}
    for yr in sorted(cohort_stats.keys()):
        s = cohort_stats[yr]
        total = s["total_papers"]
        report[yr] = {
            "total_papers": total,
            "pct_with_cby": (s["has_cby_entries"] / total * 100) if total else 0.0,
            "pct_with_window_citations": (s["has_window_citations"] / total * 100) if total else 0.0,
            "min_year_seen": s["min_cby_year"] if s["min_cby_year"] != 9999 else None,
            "max_year_seen": s["max_cby_year"] if s["max_cby_year"] != -9999 else None,
        }

    return report


def stream_jsonl(filepath: str) -> Iterable[Dict[str, Any]]:
    """Stream JSON lines from file one by one for low-memory operation."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    """Diagnostic CLI to inspect citation label coverage."""
    import argparse
    parser = argparse.ArgumentParser(description="Aurelium Citation Label Inspector")
    parser.add_argument("--jsonl", default="data/raw/papers.jsonl", help="Path to papers.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.jsonl):
        print(f"File not found: {args.jsonl}")
        return

    print(f"Checking coverage for {args.jsonl}...")
    stats = check_coverage(stream_jsonl(args.jsonl))
    print(f"{'Year':<6} | {'Total':<8} | {'% Has Cby':<12} | {'% Non-zero Window':<20} | {'Year Range in Cby'}")
    print("-" * 75)
    for yr, s in stats.items():
        yr_range = f"{s['min_year_seen']} - {s['max_year_seen']}"
        print(f"{yr:<6} | {s['total_papers']:<8} | {s['pct_with_cby']:<11.1f}% | {s['pct_with_window_citations']:<19.1f}% | {yr_range}")


if __name__ == "__main__":
    main()
