"""
Aurelium - Phase 1: OpenAlex Data Collection Module
Fetches scientific literature records from the OpenAlex Works API for Artificial Intelligence.
Streams data line-by-line to disk to support memory-constrained environments (<= 4 GB RAM).
"""

import argparse
import collections
import datetime
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_SUBFIELD_ID = "1702"  # OpenAlex subfield: Artificial Intelligence
DEFAULT_YEARS = range(2015, 2024)
DEFAULT_PER_YEAR = 5000
DEFAULT_SEED = 42
DEFAULT_PER_PAGE = 200  # Maximum page size supported by OpenAlex works endpoint


def get_auth_params() -> Dict[str, str]:
    """
    Retrieve OpenAlex authentication and polite-pool credentials.
    Checks environment variables first (e.g. from ml/.env), then falls back
    to Kaggle UserSecretsClient if running on Kaggle.
    Neither keys nor email addresses are logged or printed.
    """
    api_key = os.getenv("OPENALEX_API_KEY")
    email = os.getenv("OPENALEX_EMAIL")

    if not api_key or not email:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            secrets = UserSecretsClient()
            if not api_key:
                try:
                    api_key = secrets.get_secret("OPENALEX_API_KEY")
                except Exception:
                    pass
            if not email:
                try:
                    email = secrets.get_secret("OPENALEX_EMAIL")
                except Exception:
                    pass
        except Exception:
            pass

    params: Dict[str, str] = {}
    if api_key:
        params["api_key"] = api_key.strip()
    if email:
        params["mailto"] = email.strip()
    return params


def reconstruct_abstract(abstract: Optional[Dict[str, List[int]]]) -> Optional[str]:
    """
    Reconstruct full text abstract from OpenAlex abstract_inverted_index.
    The inverted index maps words to a list of integer token positions.
    """
    if not abstract:
        return None

    words = []
    for word, positions in abstract.items():
        for position in positions:
            words.append((position, word))

    words.sort()
    return " ".join(word for position, word in words)


def extract_record(paper: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a flattened record containing only non-leaking fields.

    Leakage prevention:
    cited_by_count, fwci, citation_normalized_percentile, cited_by_percentile_year,
    and related_works are deliberately EXCLUDED to prevent future data leakage into
    Phase 2 (citation prediction).
    counts_by_year is PRESERVED as it provides the future citation label source.
    """
    # Handle topic information safely
    primary_topic = paper.get("primary_topic")

    if primary_topic:
        topic = primary_topic.get("display_name")
        subfield_data = primary_topic.get("subfield")
        subfield = subfield_data.get("display_name") if subfield_data else None
        field_data = primary_topic.get("field")
        field = field_data.get("display_name") if field_data else None
    else:
        topic = None
        subfield = None
        field = None

    # Reconstruct abstract
    abstract_text = reconstruct_abstract(paper.get("abstract_inverted_index"))

    # Return cleaned record (12 specific fields)
    return {
        "id": paper.get("id"),
        "title": paper.get("title"),
        "abstract": abstract_text,
        "publication_date": paper.get("publication_date"),
        "publication_year": paper.get("publication_year"),
        "type": paper.get("type"),
        "language": paper.get("language"),
        "referenced_works": paper.get("referenced_works"),
        "counts_by_year": paper.get("counts_by_year"),
        "topic": topic,
        "subfield": subfield,
        "field": field,
    }


def get_with_retry(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_attempts: int = 6,
    base_delay: float = 1.0,
    timeout: float = 30.0,
) -> requests.Response:
    """
    Execute GET request with exponential backoff and jitter on 429 and 5xx errors.
    Respects Retry-After headers if present. Fails immediately on other 4xx errors.
    """
    merged_params = dict(params or {})
    # Merge authentication and polite pool credentials
    for k, v in get_auth_params().items():
        if k not in merged_params:
            merged_params[k] = v

    headers = {
        "User-Agent": "Aurelium-Research-Copilot/1.0"
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=merged_params, headers=headers, timeout=timeout)

            # Success
            if resp.status_code == 200:
                return resp

            # 429 Rate Limited or 5xx Server Error -> Backoff and retry
            if resp.status_code == 429 or 500 <= resp.status_code <= 599:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_seconds = float(retry_after)
                    except ValueError:
                        sleep_seconds = base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                else:
                    sleep_seconds = base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)

                if attempt == max_attempts:
                    resp.raise_for_status()

                time.sleep(sleep_seconds)
                continue

            # Other 4xx client errors (400, 401, 403, 404, etc.) -> Fail fast
            if 400 <= resp.status_code < 500:
                raise requests.HTTPError(
                    f"OpenAlex client error {resp.status_code}: {resp.text}",
                    response=resp,
                )

            # Any other unexpected non-200 code
            resp.raise_for_status()

        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt == max_attempts:
                raise
            sleep_seconds = base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
            time.sleep(sleep_seconds)

    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to fetch {url} after {max_attempts} attempts.")


def fetch_year_counts(
    out_csv: str,
    api_base: str = OPENALEX_WORKS_URL,
    subfield_id: str = DEFAULT_SUBFIELD_ID,
) -> Dict[int, int]:
    """
    Fetch true total paper counts per year from OpenAlex using group_by.

    Why fetch year counts separately?
    Our primary dataset draws a fixed sample of 5,000 papers per year, which
    artificially suppresses publication volume growth over time.
    These population counts provide the ground-truth volume for Phase 7 trend forecasting.
    """
    params = {
        "filter": f"primary_topic.subfield.id:{subfield_id},language:en,publication_year:2015-2023",
        "group_by": "publication_year",
    }
    resp = get_with_retry(api_base, params=params)
    data = resp.json()
    group_by_list = data.get("group_by", [])

    year_counts: Dict[int, int] = {}
    for entry in group_by_list:
        try:
            year = int(entry["key"])
            count = int(entry["count"])
            year_counts[year] = count
        except (KeyError, ValueError):
            continue

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

    # Write sorted CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("year,count\n")
        for year in sorted(year_counts.keys()):
            f.write(f"{year},{year_counts[year]}\n")

    return year_counts


def collect(
    out_jsonl: str,
    years: Any = DEFAULT_YEARS,
    per_year: int = DEFAULT_PER_YEAR,
    seed: int = DEFAULT_SEED,
    sleep: float = 0.2,
    per_page: int = DEFAULT_PER_PAGE,
    subfield_id: str = DEFAULT_SUBFIELD_ID,
    api_base: str = OPENALEX_WORKS_URL,
) -> Dict[str, Any]:
    """
    Collect sample of papers per year and stream directly to JSONL.

    Design details:
    - Why basic page paging instead of cursor?
      OpenAlex does not support cursor paging when using the 'sample' parameter.
      Sample creates a pseudo-randomized stream indexed by page numbers (1..N).
    - Why a fixed seed?
      The seed guarantees reproducible sampling across repeated queries and paging calls.
    - Resumable:
      Scans existing output file for existing IDs and yearly totals on startup.
      Skips years that are already complete, and avoids appending duplicate IDs.
    - Crash-tolerant:
      Streams and flushes each page to disk, ignoring any partially written lines.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)

    seen_ids: Set[str] = set()
    years_list = sorted(list(years))
    year_counts: Dict[int, int] = {y: 0 for y in years_list}
    duplicates_skipped = 0
    missing_abstracts = 0

    # Resume logic: read existing file line by line
    if os.path.exists(out_jsonl):
        print(f"Reading existing file for resume: {out_jsonl}")
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore corrupted/truncated trailing line from a crash
                    continue

                rec_id = record.get("id")
                if rec_id:
                    seen_ids.add(rec_id)

                rec_year = record.get("publication_year")
                if rec_year in year_counts:
                    year_counts[rec_year] += 1

                if not record.get("abstract"):
                    missing_abstracts += 1

        print(f"Existing records: {len(seen_ids)} | Per-year status: {year_counts}")

    # Open file in append mode to stream records
    with open(out_jsonl, "a", encoding="utf-8") as out_file:
        for year in years_list:
            if year_counts[year] >= per_year:
                print(f"Year {year} already complete ({year_counts[year]}/{per_year} records). Skipping.")
                continue

            max_pages = math.ceil(per_year / per_page)
            print(f"\n--- Collecting Year {year}: Target {per_year} (up to {max_pages} pages) ---")

            for page in range(1, max_pages + 1):
                if year_counts[year] >= per_year:
                    break

                params = {
                    "filter": f"primary_topic.subfield.id:{subfield_id},language:en,publication_year:{year}",
                    "sample": per_year,
                    "seed": seed,
                    "per-page": per_page,
                    "page": page,
                    "select": "id,title,abstract_inverted_index,publication_date,publication_year,type,language,referenced_works,counts_by_year,primary_topic",
                }

                resp = get_with_retry(api_base, params=params)
                page_data = resp.json()
                results = page_data.get("results", [])

                if not results:
                    print(f"Year {year} page {page}: No more results returned by API.")
                    break

                new_written_page = 0
                abstracts_on_page = 0

                for paper in results:
                    paper_id = paper.get("id")
                    if not paper_id:
                        continue

                    if paper_id in seen_ids:
                        duplicates_skipped += 1
                        continue

                    if year_counts[year] >= per_year:
                        break

                    record = extract_record(paper)
                    if record.get("abstract"):
                        abstracts_on_page += 1
                    else:
                        missing_abstracts += 1

                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    seen_ids.add(paper_id)
                    year_counts[year] += 1
                    new_written_page += 1

                # Flush after each page to persist data safely against crashes
                out_file.flush()

                running_total = sum(year_counts.values())
                print(
                    f"Year {year} | Page {page:2d} | Records: {len(results):3d} | "
                    f"New: {new_written_page:3d} | Year Total: {year_counts[year]}/{per_year} | "
                    f"Running Total: {running_total} | Abstracts: {abstracts_on_page}/{len(results)}"
                )

                if sleep > 0:
                    time.sleep(sleep)

                if len(results) < per_page:
                    break

    stats = {
        "year_counts": year_counts,
        "total": sum(year_counts.values()),
        "seen_ids_count": len(seen_ids),
        "duplicates_skipped": duplicates_skipped,
        "missing_abstracts": missing_abstracts,
    }
    return stats


def write_manifest(
    out_json: str,
    stats: Dict[str, Any],
    years: Any,
    per_year: int,
    seed: int,
    subfield_id: str = DEFAULT_SUBFIELD_ID,
    api_base: str = OPENALEX_WORKS_URL,
) -> None:
    """
    Write data collection provenance manifest.
    Contains query specifications, environment versions, and run statistics.
    Explicitly excludes sensitive credentials (email, api keys).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)

    manifest_data = {
        "api_base_url": api_base,
        "filter": {
            "primary_topic.subfield.id": subfield_id,
            "language": "en",
            "publication_years": sorted(list(years)),
        },
        "sampling": {
            "target_per_year": per_year,
            "seed": seed,
            "sampling_method": "OpenAlex sample parameter with page paging",
        },
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "counts_per_year": stats.get("year_counts", {}),
        "total_records": stats.get("total", 0),
        "duplicates_skipped": stats.get("duplicates_skipped", 0),
        "missing_abstracts_count": stats.get("missing_abstracts", 0),
        "environment": {
            "python_version": sys.version,
            "requests_version": requests.__version__,
        },
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)


def data_report(jsonl_path: str, out_md_path: str) -> None:
    """
    Generate an inspection report from collected JSONL data.
    Streams line-by-line to run with minimal RAM footprint (< 4 GB constraint).
    Saves markdown summary with totals, topic distributions, and missingness rates.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_md_path)), exist_ok=True)

    total_records = 0
    records_per_year: Dict[int, int] = collections.defaultdict(int)
    missing_abstracts_overall = 0
    missing_abstracts_per_year: Dict[int, int] = collections.defaultdict(int)
    missing_topic_count = 0
    topic_counts: Dict[str, int] = collections.defaultdict(int)
    type_counts: Dict[str, int] = collections.defaultdict(int)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_records += 1
            year = rec.get("publication_year")
            if year is not None:
                records_per_year[int(year)] += 1

            if not rec.get("abstract"):
                missing_abstracts_overall += 1
                if year is not None:
                    missing_abstracts_per_year[int(year)] += 1

            topic = rec.get("topic")
            if topic:
                topic_counts[topic] += 1
            else:
                missing_topic_count += 1

            doc_type = rec.get("type") or "unknown"
            type_counts[doc_type] += 1

    # Format Markdown Report
    lines = [
        "# Aurelium - Raw Data Inspection Report",
        "",
        f"**Source File**: `{os.path.basename(jsonl_path)}`  ",
        f"**Generated At (UTC)**: `{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`  ",
        f"**Total Records Analyzed**: `{total_records:,}`  ",
        "",
        "---",
        "",
        "## 1. Summary Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total Papers | {total_records:,} |",
        f"| Missing Abstract % | {(missing_abstracts_overall / total_records * 100):.2f}% ({missing_abstracts_overall:,} / {total_records:,}) |" if total_records else "| Missing Abstract % | 0.00% |",
        f"| Missing Primary Topic % | {(missing_topic_count / total_records * 100):.2f}% ({missing_topic_count:,} / {total_records:,}) |" if total_records else "| Missing Primary Topic % | 0.00% |",
        f"| Unique Topics Identified | {len(topic_counts):,} |",
        "",
        "## 2. Records and Missingness by Publication Year",
        "",
        "| Year | Total Papers | Missing Abstracts | Missing Abstract % |",
        "|---|---|---|---|",
    ]

    for yr in sorted(records_per_year.keys()):
        yr_total = records_per_year[yr]
        yr_missing = missing_abstracts_per_year[yr]
        yr_pct = (yr_missing / yr_total * 100) if yr_total > 0 else 0.0
        lines.append(f"| {yr} | {yr_total:,} | {yr_missing:,} | {yr_pct:.2f}% |")

    lines.extend([
        "",
        "## 3. Publication Types Breakdown",
        "",
        "| Type | Count | Percentage |",
        "|---|---|---|",
    ])
    for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_records * 100) if total_records else 0.0
        lines.append(f"| `{doc_type}` | {count:,} | {pct:.2f}% |")

    lines.extend([
        "",
        "## 4. Top 20 Primary Topics",
        "",
        "| Rank | Topic | Count | Percentage |",
        "|---|---|---|---|",
    ])
    top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    for rank, (topic_name, count) in enumerate(top_topics, 1):
        pct = (count / total_records * 100) if total_records else 0.0
        lines.append(f"| {rank} | {topic_name} | {count:,} | {pct:.2f}% |")

    report_content = "\n".join(lines) + "\n"
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def main() -> None:
    """CLI entry point for data collection."""
    parser = argparse.ArgumentParser(description="Aurelium OpenAlex Data Collector")
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Target directory for output files (default: data)",
    )
    parser.add_argument(
        "--years",
        default="2015-2023",
        help="Range or list of years (e.g. '2015-2023', '2020', '2019,2020')",
    )
    parser.add_argument(
        "--per-year",
        type=int,
        default=DEFAULT_PER_YEAR,
        help=f"Target random sample size per year (default: {DEFAULT_PER_YEAR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for sampling reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between API requests (default: 0.2)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a lightweight 1-year x 1-page test fetch (10 papers)",
    )

    args = parser.parse_args()

    # Determine years to collect
    if args.test:
        years = [2020]
        per_year = 10
        out_dir = os.path.join(args.out_dir, "test")
        per_page = 10
    else:
        out_dir = args.out_dir
        per_year = args.per_year
        per_page = DEFAULT_PER_PAGE
        if "-" in args.years:
            start_y, end_y = map(int, args.years.split("-"))
            years = list(range(start_y, end_y + 1))
        elif "," in args.years:
            years = [int(y.strip()) for y in args.years.split(",")]
        else:
            years = [int(args.years.strip())]

    raw_dir = os.path.join(out_dir, "raw")
    reports_dir = os.path.join(out_dir, "reports") if not args.test else os.path.join(out_dir, "reports")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    counts_csv = os.path.join(raw_dir, "year_counts.csv")
    papers_jsonl = os.path.join(raw_dir, "papers.jsonl")
    manifest_json = os.path.join(raw_dir, "manifest.json")
    report_md = os.path.join(reports_dir, "data_report.md")

    print(f"=== Aurelium OpenAlex Collector {'(TEST MODE)' if args.test else ''} ===")
    print(f"Output directory : {out_dir}")
    print(f"Target years     : {years}")
    print(f"Per-year target  : {per_year}")
    print(f"Sampling seed    : {args.seed}")
    print(f"Sleep interval   : {args.sleep}s")

    # Step 1: Fetch total year counts (Phase 7 ground truth)
    print("\n[Step 1/4] Fetching publication year population counts...")
    counts = fetch_year_counts(counts_csv)
    print(f"Saved population counts to {counts_csv}: {counts}")

    # Step 2: Sample and stream records to JSONL
    print("\n[Step 2/4] Sampling and streaming papers to JSONL...")
    stats = collect(
        out_jsonl=papers_jsonl,
        years=years,
        per_year=per_year,
        seed=args.seed,
        sleep=args.sleep,
        per_page=per_page,
    )

    # Step 3: Write provenance manifest
    print("\n[Step 3/4] Writing provenance manifest...")
    write_manifest(
        out_json=manifest_json,
        stats=stats,
        years=years,
        per_year=per_year,
        seed=args.seed,
    )
    print(f"Saved manifest to {manifest_json}")

    # Step 4: Generate data inspection report
    print("\n[Step 4/4] Generating data inspection report...")
    data_report(jsonl_path=papers_jsonl, out_md_path=report_md)
    print(f"Saved data report to {report_md}")

    print("\nCollection pipeline finished successfully!")


if __name__ == "__main__":
    main()
