"""
Aurelium - Phase 3: Metadata Enrichment
Fetches fields Phase 1 did not collect (DOI, retraction/paratext flags, venue,
author ids) for papers already in papers.jsonl, keyed by OpenAlex work id.
Writes to a separate enrichment.jsonl so raw papers.jsonl is never touched.

Leakage prevention:
cited_by_count, fwci, citation_normalized_percentile, cited_by_percentile_year
and related_works are deliberately NOT selected (see README.md).
"""

import argparse
import json
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Set

from src.citation_labels import stream_jsonl
from src.openalex_collect import OPENALEX_WORKS_URL, get_auth_params, get_with_retry

DEFAULT_BATCH_SIZE = 50  # OpenAlex allows up to 50 values in one OR-filter
SELECT_FIELDS = "id,doi,is_retracted,is_paratext,primary_location,authorships"


def short_id(work_id: str) -> str:
    """Turn 'https://openalex.org/W123' into 'W123' for use in a filter."""
    return work_id.rsplit("/", 1)[-1]


def flatten(work: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten one OpenAlex work into an enrichment record.
    Keeps author ids only, never names. Missing nested objects become None.
    """
    location = work.get("primary_location") or {}
    source = location.get("source") or {}

    author_ids: List[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        if author.get("id"):
            author_ids.append(author["id"])

    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "is_retracted": work.get("is_retracted"),
        "is_paratext": work.get("is_paratext"),
        "venue": source.get("display_name"),
        "venue_id": source.get("id"),
        "venue_type": source.get("type"),
        "author_ids": author_ids,
        "n_authors": len(work.get("authorships") or []),
    }


def fetch_batch(ids: List[str], api_base: str = OPENALEX_WORKS_URL) -> List[Dict[str, Any]]:
    """Fetch up to 50 works in a single API call using an OR-filter on ids."""
    params = {
        "filter": "openalex:" + "|".join(short_id(i) for i in ids),
        "select": SELECT_FIELDS,
        "per-page": DEFAULT_BATCH_SIZE,
    }
    params.update(get_auth_params())
    resp = get_with_retry(api_base, params=params)
    return resp.json().get("results", [])


def load_done_ids(out_jsonl: str) -> Set[str]:
    """Read ids already enriched, so a restarted run skips them."""
    if not os.path.exists(out_jsonl):
        return set()
    return {rec["id"] for rec in stream_jsonl(out_jsonl) if rec.get("id")}


def ensure_trailing_newline(path: str) -> None:
    """If a crash left a half-written last line, end it so new lines start clean."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
    if last != b"\n":
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")


def fetch_metadata(
    ids: Iterable[str],
    out_jsonl: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    sleep: float = 0.2,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Enrich ids in batches, appending each batch to out_jsonl and flushing.
    Ids already in out_jsonl are skipped. Returns counts and the ids OpenAlex did not return.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    done = load_done_ids(out_jsonl)
    todo = [i for i in dict.fromkeys(ids) if i not in done]  # dedupe, keep order
    print(f"Already enriched: {len(done)} | To fetch: {len(todo)}")

    ensure_trailing_newline(out_jsonl)
    written = 0
    missing: List[str] = []

    with open(out_jsonl, "a", encoding="utf-8") as out_file:
        for n, start in enumerate(range(0, len(todo), batch_size), 1):
            if max_batches is not None and n > max_batches:
                break
            batch = todo[start:start + batch_size]
            works = fetch_batch(batch)

            returned: Set[str] = set()
            for work in works:
                record = flatten(work)
                if record["id"] in done or record["id"] not in batch:
                    continue
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                done.add(record["id"])
                returned.add(record["id"])
                written += 1
            out_file.flush()  # persist each batch so a crash loses at most one call

            missing.extend(i for i in batch if i not in returned)
            print(f"Batch {n} | Asked: {len(batch)} | Returned: {len(returned)} | Written total: {written}")

            if sleep > 0:
                time.sleep(sleep)

    return {"written": written, "skipped_existing": len(done) - written, "missing_ids": missing}


def main() -> None:
    """CLI entry point for metadata enrichment."""
    parser = argparse.ArgumentParser(description="Aurelium OpenAlex Metadata Enrichment")
    parser.add_argument("--jsonl", default="data/raw/papers.jsonl", help="Input papers.jsonl (read only)")
    parser.add_argument("--out", default="data/raw/enrichment.jsonl", help="Output enrichment.jsonl")
    parser.add_argument("--sleep", type=float, default=0.2, help="Seconds between API calls")
    parser.add_argument("--test", action="store_true", help="Enrich data/test papers with a single API call")
    args = parser.parse_args()

    if args.test:
        args.jsonl = "data/test/raw/papers.jsonl"
        args.out = "data/test/raw/enrichment.jsonl"

    ids = [rec["id"] for rec in stream_jsonl(args.jsonl) if rec.get("id")]
    print(f"=== Aurelium Enrichment {'(TEST MODE)' if args.test else ''} ===")
    print(f"Input : {args.jsonl} ({len(ids)} ids)")
    print(f"Output: {args.out}")

    stats = fetch_metadata(ids, args.out, sleep=args.sleep, max_batches=1 if args.test else None)

    print(f"\nWritten this run: {stats['written']}")
    print(f"Ids OpenAlex did not return: {len(stats['missing_ids'])}")
    for missing_id in stats["missing_ids"][:20]:
        print(f"  {missing_id}")


if __name__ == "__main__":
    main()
