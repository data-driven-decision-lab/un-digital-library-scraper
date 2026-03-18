# Database Schema Reference

This document describes every table in the UN Digital Library project's Turso database.
The database engine is [LibSQL](https://github.com/tursodatabase/libsql) (Turso),
which is wire-compatible with SQLite. All DDL is standard SQLite syntax.

**Authoritative DDL:** `db/schema.sql`

Apply schema changes with:
```bash
cat db/schema.sql | turso db shell unga-datadrivendecisionlab
```

---

## Tables

1. [un_votes_raw](#1-un_votes_raw)
2. [un_votes_with_sc](#2-un_votes_with_sc)
3. [annual_scores](#3-annual_scores)
4. [topic_votes_yearly](#4-topic_votes_yearly)
5. [pairwise_similarity_yearly](#5-pairwise_similarity_yearly)
6. [pipeline_runs](#6-pipeline_runs)

---

## 1. `un_votes_raw`

**Purpose:** Raw scraper output before Security Council tagging. One row per UN General Assembly
resolution as scraped from [digitallibrary.un.org](https://digitallibrary.un.org).

**Populated by:** `scraper_pipeline.py`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-increment primary key |
| `Resolution` | TEXT | Resolution identifier (e.g., `A/RES/79/1`) |
| `Date` | TEXT | Voting date in ISO format (`YYYY-MM-DD`) |
| `Title` | TEXT | Full resolution title text |
| `Link` | TEXT UNIQUE | Source URL on digitallibrary.un.org |
| `tags` | TEXT | Comma-separated UNBIS subject tags (Main Category and Subcategory level) |
| `vote_data` | TEXT | JSON blob — see **vote_data format** below |

### vote_data format

```json
{
  "AFG": "YES",
  "ALB": "NO",
  "DZA": "ABSTAIN",
  "AND": null,
  ...
}
```

Keys are ISO 3166-1 alpha-3 country codes (approximately 190 countries). Values are:

| Value | Meaning |
|---|---|
| `"YES"` | Country voted yes |
| `"NO"` | Country voted no |
| `"ABSTAIN"` | Country abstained |
| `null` | Country did not participate (non-member or non-voting at the time) |

The dashboard pipeline expands this blob into per-country columns at load time via
`_expand_vote_data()`. Storing votes as JSON avoids enumerating all 190+ country
columns in the DDL while keeping a single row per resolution.

---

## 2. `un_votes_with_sc`

**Purpose:** Enriched voting data including Security Council resolutions.
This is the **primary source table** for the dashboard data pipeline.

**Populated by:** `scraper_pipeline.py` after SC tagging

Schema is identical to `un_votes_raw` with one additional column:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-increment primary key |
| `Resolution` | TEXT | Resolution identifier |
| `Date` | TEXT | Voting date in ISO format (`YYYY-MM-DD`) |
| `Title` | TEXT | Full resolution title text |
| `Link` | TEXT UNIQUE | Source URL on digitallibrary.un.org |
| `tags` | TEXT | Comma-separated UNBIS subject tags |
| `vote_data` | TEXT | JSON blob — same format as `un_votes_raw.vote_data` |
| `sc_flag` | INTEGER DEFAULT 0 | `1` if Security Council resolution (Resolution starts with `"S/"`), else `0` |

**Filtering in the pipeline:** `dashboard_data_pipeline.py` excludes rows where
`Resolution LIKE 'S/%'` before computing any scores. The `sc_flag` column makes
this filtering explicit and queryable without a string-prefix scan.

---

## 3. `annual_scores`

**Purpose:** Per-country, per-year pillar scores, total index, ranks, and vote counts.
This is the primary table consumed by the API for report and rankings endpoints.

**Populated by:** `dashboard_data_pipeline.py` (`generate_combined_index` + `generate_annual_scores`)

**Unique constraint:** `(Year, Country)`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-increment primary key |
| `Year` | INTEGER | Calendar year of the votes |
| `Country` | TEXT | ISO 3166-1 alpha-3 country code |
| `Pillar 1 Score` | REAL | Min-max normalized Pillar 1 score (0–100). **Equal to `Pillar 1 Normalized`** — see note below |
| `Pillar 2 Score` | REAL | Min-max normalized Pillar 2 score (0–100). **Equal to `Pillar 2 Normalized`** |
| `Pillar 3 Score` | REAL | Min-max normalized Pillar 3 score (0–100). **Equal to `Pillar 3 Normalized`** |
| `Total Index Average` | REAL | Mean of the three normalized pillar scores (0–100) |
| `Overall Rank` | INTEGER | Rank within the year, descending by `Total Index Average` (rank 1 = best) |
| `Overall Rank Rolling Avg (3y)` | REAL | 3-year rolling mean of `Overall Rank` per country |
| `Total Index Normalized` | REAL | Direct copy of `Total Index Average` — no second normalization |
| `Pillar 1 Normalized` | REAL | Min-max normalized P1 score (0–100) |
| `Pillar 1 Rank` | INTEGER | Rank within the year by raw P1 score (before normalization) |
| `Pillar 2 Normalized` | REAL | Min-max normalized P2 score (0–100) |
| `Pillar 2 Rank` | INTEGER | Rank within the year by raw P2 score |
| `Pillar 3 Normalized` | REAL | Min-max normalized P3 score (0–100) |
| `Pillar 3 Rank` | INTEGER | Rank within the year by raw P3 score |
| `Yes Votes` | INTEGER | Total YES votes cast by this country in this year |
| `No Votes` | INTEGER | Total NO votes cast by this country in this year |
| `Abstain Votes` | INTEGER | Total ABSTAIN votes cast by this country in this year |
| `Total Votes in Year` | INTEGER | Sum of Yes + No + Abstain votes |

**Important — Score vs. Normalized columns:**
"Pillar X Score" and "Pillar X Normalized" hold the **same value** (PIPE-06).
Both columns store the min-max normalized score for the year cohort. The raw
pre-normalization computation output is not persisted.

---

## 4. `topic_votes_yearly`

**Purpose:** Per-country, per-year, per-topic vote counts. Used for topic analysis
and breakdown charts in reports.

**Populated by:** `dashboard_data_pipeline.py` (`generate_topic_votes()`)

**Unique constraint:** `(Year, Country, TopicTag)`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-increment primary key |
| `Year` | INTEGER | Calendar year |
| `Country` | TEXT | ISO 3166-1 alpha-3 country code |
| `TopicTag` | TEXT | UNBIS Main Category or Subcategory name (e.g., `"DISARMAMENT"`, `"Nuclear weapons"`) |
| `YesVotes_Topic` | INTEGER DEFAULT 0 | YES votes cast by this country on resolutions tagged with this topic in this year |
| `NoVotes_Topic` | INTEGER DEFAULT 0 | NO votes cast |
| `AbstainVotes_Topic` | INTEGER DEFAULT 0 | ABSTAIN votes cast |
| `TotalVotes_Topic` | INTEGER DEFAULT 0 | Sum of Yes + No + Abstain votes for this topic |

A single resolution can contribute to multiple topic tags if it has multiple UNBIS
tags. The pipeline uses `parse_tags_for_subtag1()`, which matches both Main Category
and Subcategory tags from the `un_classification` dictionary.

---

## 5. `pairwise_similarity_yearly`

**Purpose:** Cosine similarity between every pair of countries for each year. Used
for ally/adversary analysis and regional alignment visualisations.

**Populated by:** `dashboard_data_pipeline.py` (`generate_similarity_matrix()`)

**Unique constraint:** `(Year, Country1, Country2)`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-increment primary key |
| `Year` | INTEGER | Calendar year |
| `Country1` | TEXT | ISO 3166-1 alpha-3 code of the alphabetically-first country in the pair |
| `Country2` | TEXT | ISO 3166-1 alpha-3 code of the alphabetically-second country in the pair |
| `CosineSimilarity` | REAL | Cosine similarity of the two countries' encoded vote vectors, range −1.0 to +1.0 |

**Pair ordering:** Only pairs where `Country1 < Country2` (lexicographic order) are stored.
This prevents duplicate rows for the same country pair. To query the alignment between
two specific countries, always pass the alphabetically lower ISO3 code as `Country1`.

Example: to find the similarity between `USA` and `CHN`, query `Country1 = 'CHN'`
and `Country2 = 'USA'` (because `'CHN' < 'USA'`).

**Precision:** `CosineSimilarity` is stored at full float precision with no rounding.
Values range from −1.0 (perfectly opposing votes) to +1.0 (identical votes). A value
of 0 indicates no linear relationship in the encoded vote vectors.

---

## 6. `pipeline_runs`

**Purpose:** Execution metadata for each pipeline run. Enables monitoring, debugging,
and auditing of both the scraper and dashboard pipelines.

**Populated by:** both `scraper_pipeline.py` and `dashboard_data_pipeline.py`

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-increment primary key |
| `run_id` | TEXT UNIQUE | UUID v4 generated at pipeline start |
| `pipeline_name` | TEXT | `"scraper_pipeline"` or `"dashboard_data_pipeline"` |
| `started_at` | TEXT | ISO datetime when the run began (UTC) |
| `finished_at` | TEXT | ISO datetime when the run completed or failed (null if still running) |
| `status` | TEXT DEFAULT `'running'` | `"running"` → `"success"` or `"failed"` |
| `rows_affected` | INTEGER DEFAULT 0 | Total rows upserted across all output tables |
| `error_message` | TEXT | Error traceback or message if `status = "failed"`, else null |
| `notes` | TEXT | JSON blob for pipeline-specific metrics (e.g., scraper progress counters) |
