# Aurelium - Leakage Table (Phase 3)

**Prediction moment:** the paper's publication date. A column may be a feature only if its value
could have been known on that day. Labels come from citations in `publication_year .. publication_year + 2`.

**Verdicts:** **Yes** = allowed feature. **Caution** = allowed, but the stored value may have been updated
after publication; check before relying on it. **No** = not a feature (identifier, audit copy, pipeline
bookkeeping or future information). **Label-only** = used to build the target, never as input.
**Banned** = excluded at collection (README.md) and must never be added.

## Columns in `clean_papers.parquet`

| Column | Known at publication? | Feature? | Reason |
|---|---|---|---|
| `id` | Yes | No | Identifier; a model would memorize it, not learn from it |
| `split` | Yes | No | Bookkeeping derived from year; decides which rows train/val/test |
| `publication_year` | Yes | No | Defines the split; test years are never seen in training, so a model cannot use it fairly |
| `publication_date` | Yes | Yes | Phase 2 uses publication month; ignore it when `date_is_imputed` (July 1 is a fill value) |
| `date_is_imputed` | Yes | No | Data-quality flag about our pipeline, not about the paper |
| `date_year_mismatch` | Yes | No | Data-quality flag |
| `type` | Yes | Yes | Used in Phase 2; a preprint may later be relabelled article, see venue caution |
| `language` | Yes | No | Constant (`en`) after filtering, so it carries no signal |
| `title` | Yes | Yes | Embedding input (SPECTER) |
| `abstract` | Yes | Yes | Embedding input; title fallback when missing |
| `title_raw` | Yes | No | Audit copy of `title` |
| `abstract_raw` | Yes | No | Audit copy of `abstract` |
| `has_abstract` | Mostly | Caution | Used in Phase 2; some publishers deposit abstracts years later, so OpenAlex today may know more than on publication day |
| `abstract_word_count` | Mostly | Caution | Same reason as `has_abstract` |
| `abstract_short` | Mostly | Caution | Same reason as `has_abstract` |
| `abstract_boilerplate` | Mostly | Caution | Same reason as `has_abstract` |
| `doi` | Yes | No | Identifier |
| `doi_norm` | Yes | No | Identifier (dedup key) |
| `venue` | Not always | Caution | `primary_location` can be updated after a preprint is accepted; "accepted at NeurIPS" is a future event that predicts citations |
| `venue_canonical` | Not always | Caution | Same reason as `venue` |
| `venue_id` | Not always | Caution | Same reason as `venue` |
| `venue_type` | Not always | Caution | Same reason as `venue` (repository -> conference after acceptance) |
| `author_ids` | Yes | No (raw) | Raw ids are identifiers. Author features (h-index, past citations) are allowed only if computed from papers before this paper's date; today's h-index includes this paper's own future citations |
| `n_authors` | Yes | Yes | Team size is fixed at publication |
| `is_retracted` | No | No | Retraction happens after publication; also filtered, so constant |
| `is_paratext` | Yes | No | Filtered, so constant |
| `has_enrichment` | n/a | No | Pipeline bookkeeping |
| `topic` | Mostly | Caution | Used in Phase 2. Assigned recently by an OpenAlex model with a taxonomy built from today's literature, so topic names reflect which areas later became popular |
| `subfield` | Mostly | Caution | Same reason as `topic`; near-constant here because we sampled one subfield |
| `field` | Mostly | Caution | Same reason as `topic` |
| `referenced_works` | Yes | Yes | A paper's own reference list is fixed at publication; Phase 2 uses its length |
| `citations_3yr` | No | Label-only | The target |
| `log_citations_3yr` | No | Label-only | The target, log1p-transformed |
| `label_usable` | No | No | Row filter for training, derived from citation coverage |
| `window_truncated` | No | No | Row filter for training, derived from citation coverage |
| `pre_pub_citations` | Partly | No | Derived from `counts_by_year` (the label source); citations before the publication year exist, but OpenAlex attributes them retrospectively when it merges preprint and journal records |
| `impossible_citations` | No | No | Data-quality flag derived from the label source |

## Columns not in the parquet

| Column | Known at publication? | Feature? | Reason |
|---|---|---|---|
| `counts_by_year` | No | Label-only | Citations per year up to today; source of `citations_3yr`. Kept in raw only, deliberately left out of the clean file |
| `cited_by_count` | No | Banned | Total citations today: the label plus every later year |
| `fwci` | No | Banned | Field-weighted citation impact, computed from future citations |
| `citation_normalized_percentile` | No | Banned | Citation rank among peers, computed from future citations |
| `cited_by_percentile_year` | No | Banned | Citation percentile within the year, computed from future citations |
| `related_works` | No | Banned | Computed by OpenAlex against today's corpus, including papers published after this one |

## Open questions

1. Should `venue*` be a feature at all, or only `venue_type` restricted to papers whose type is not `preprint`?
2. `topic`/`subfield`/`field` are already Phase 2 features. Keep them (Caution) or rerun Phase 2 without them to measure how much they contribute?
3. `pre_pub_citations`: keep as No, or allow it as "preprint popularity before formal publication"?
