"""
Aurelium - Phase 3: tests for src.enrich.flatten
Uses hand-made OpenAlex-shaped dicts. No network calls.
"""

from src.enrich import flatten, short_id

FULL_WORK = {
    "id": "https://openalex.org/W1",
    "doi": "https://doi.org/10.1000/xyz",
    "is_retracted": False,
    "is_paratext": False,
    "primary_location": {
        "source": {
            "id": "https://openalex.org/S9",
            "display_name": "Journal of Tests",
            "type": "journal",
        }
    },
    "authorships": [
        {"author": {"id": "https://openalex.org/A1", "display_name": "Ada Example"}},
        {"author": {"id": None, "display_name": "No Id Person"}},
        {"author": {"id": "https://openalex.org/A2", "display_name": "Bob Example"}},
    ],
}


def test_flatten_full_work() -> None:
    """A complete work maps every field and keeps only non-null author ids."""
    rec = flatten(FULL_WORK)
    assert rec == {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.1000/xyz",
        "is_retracted": False,
        "is_paratext": False,
        "venue": "Journal of Tests",
        "venue_id": "https://openalex.org/S9",
        "venue_type": "journal",
        "author_ids": ["https://openalex.org/A1", "https://openalex.org/A2"],
        "n_authors": 3,
    }


def test_flatten_never_keeps_names() -> None:
    """Author names must not leak into the record."""
    assert "Ada Example" not in str(flatten(FULL_WORK))


def test_flatten_null_primary_location() -> None:
    """primary_location: null gives None venue fields instead of crashing."""
    work = {"id": "https://openalex.org/W2", "primary_location": None, "authorships": []}
    rec = flatten(work)
    assert rec["venue"] is None
    assert rec["venue_id"] is None
    assert rec["venue_type"] is None
    assert rec["author_ids"] == []
    assert rec["n_authors"] == 0


def test_flatten_location_without_source() -> None:
    """A location whose source is null (common for repository copies) also gives None."""
    work = {"id": "https://openalex.org/W3", "primary_location": {"source": None}}
    rec = flatten(work)
    assert rec["venue"] is None
    assert rec["doi"] is None
    assert rec["n_authors"] == 0


def test_short_id() -> None:
    """Full OpenAlex URLs become bare ids for the OR-filter."""
    assert short_id("https://openalex.org/W123") == "W123"
