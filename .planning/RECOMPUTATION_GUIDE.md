# Pipeline Recomputation Guide

> **Date**: 2026-03-18
> **For**: Engineer responsible for the scraper + aggregation pipeline
> **Context**: Automated audit of Supabase tables found critical data quality issues that require a pipeline re-run and targeted fixes before downstream analysis (2025 report) can proceed.
> **Audit scripts**: `analysis/4C/pipeline_audit.py`, `analysis/4C/supabase_audit.py`, `analysis/4C/supabase_deep_check.py`, `analysis/4C/validate_pairwise_pipeline.py`

---

## Executive Summary

The Supabase database has **two clean tables** (`annual_scores`, `un_votes_raw`/`un_votes_with_sc`) and **two corrupted tables** (`topic_votes_yearly`, `pairwise_similarity_yearly`) for 2025 data. There is also one test record that needs removal.

**Before recomputing**, fix the source data issues (Items A–B below). **Then** re-run the full pipeline to regenerate all three downstream tables.

---

## Part 1: Source Data Fixes (Do These First)

### A. Delete test resolution from `un_votes_raw`

**Problem**: Resolution `A/RES/79/125` titled "Test Resolution Data Type Fix" is a junk record.

**Evidence**:
- Only 3 countries voted on it (vs 130–192 for real resolutions)
- Dated 2025-01-15
- Tags contain `test` and `data-type-fix`
- Already absent from `un_votes_with_sc` (the pipeline's actual source table), which is why `annual_scores` correctly shows max 192 votes — but it still sits in `un_votes_raw`

**Fix**:
```sql
DELETE FROM un_votes_raw WHERE "Resolution" = 'A/RES/79/125';
```

**Verification**: `SELECT COUNT(*) FROM un_votes_raw WHERE "Scrape_Year" = 2025` should return 192.

---

### B. Clean junk tags from `un_votes_raw`

**Problem**: Two resolutions have debugging artifacts in their `tags` column.

| Tag | Resolutions affected |
|-----|---------------------|
| `test` | 1 |
| `data-type-fix` | 1 |
| `""` (empty string) | 3 |

**Fix**: Strip these from the `tags` column for the affected rows. If they're the same resolution as `A/RES/79/125`, they'll be removed by Fix A. Otherwise, remove just the junk tag entries from the comma-separated list.

**Verification**: No tag in the `tags` column should match `test`, `data-type-fix`, or be an empty string.

---

### C. Verify `un_votes_with_sc` is complete

**Problem**: This is the pipeline's actual source table. It currently has 192 GA resolutions for 2025 (correct — no test resolution), but we need to confirm that all real resolutions and all country columns are present and that votes match `un_votes_raw` (minus the test record).

**Verification**:
```sql
-- Should return 192 GA resolutions for 2025
SELECT COUNT(*) FROM un_votes_with_sc
WHERE extract(year from "Date") = 2025
  AND "Resolution" NOT LIKE 'S/%';

-- Country column count should be 193
-- (includes AFG and VEN even though they have all nulls)
```

---

## Part 2: Pipeline Re-run

After fixing source data, re-run the full pipeline (`main()` in the pipeline script). This regenerates all three downstream tables:

1. `annual_scores` — from `generate_combined_index()` + `generate_annual_scores()`
2. `topic_votes_yearly` — from `generate_topic_votes()`
3. `pairwise_similarity_yearly` — from `generate_similarity_matrix()`

### Pre-run checklist

- [ ] `A/RES/79/125` deleted from `un_votes_raw`
- [ ] Junk tags cleaned
- [ ] `un_votes_with_sc` has 192 GA resolutions for 2025
- [ ] `un_classification` dictionary is importable and up to date
- [ ] Region mapping CSV is present at the expected path

---

## Part 3: Issue-by-Issue Root Cause & Fix

### Issue 1 — 🔴 CRITICAL: 240/268 tags dropped in `topic_votes_yearly`

**What we found**: `un_votes_raw.tags` contains 268 unique tags for 2025 (across Main Category, Subcategory, and Specific Item levels of the UNBIS hierarchy). After aggregation, `topic_votes_yearly` contains only **29 tags** — a loss of 240 tags (89.6%).

**Missing tags include** major categories like DISARMAMENT AND MILITARY QUESTIONS (38 resolutions), PEACE (37), POLITICAL CONDITIONS (55), INSTITUTIONS (55), SOCIAL CONDITIONS AND EQUITY (48), DECOLONIZATION (17), SUSTAINABLE DEVELOPMENT (13), and 233 more.

**Root cause** (from source code review):

The `generate_topic_votes()` function uses `parse_tags_for_subtag1()` which walks the `tags` string looking for items that appear as **keys in `un_classification`** (Main Category level), then looks for the next item that appears as a **sub-key** (Subcategory level). If no Subcategory match is found, it falls back to the Main Category key. If no Main Category key is found at all, tags are dropped entirely.

The issue is that the `tags` column in `un_votes_raw` stores a **flat comma-separated list** of tags from all three UNBIS hierarchy levels mixed together (e.g., `"POLITICAL AND LEGAL QUESTIONS, DISARMAMENT AND MILITARY QUESTIONS, NUCLEAR DISARMAMENT, DISARMAMENT, ARMS LIMITATION"`). The parser walks this left-to-right looking for `main → sub` pairs, but:
1. **Tags that don't match any `un_classification` key are silently dropped.** If the dictionary doesn't have an entry for `DISARMAMENT AND MILITARY QUESTIONS`, that entire category vanishes.
2. **The parser returns at most one tag per Main Category per resolution.** If a resolution has multiple Subcategories under the same Main Category, only the first match survives.
3. **If the dictionary is out of date or missing entries**, tags that exist in the raw data won't match.

**Suggested fix**:
1. **Audit `un_classification`** — compare its keys against the 268 tags found in `un_votes_raw`. Print which raw tags have no dictionary match. This will immediately show if the dictionary is incomplete.
2. **Decide on tag granularity** — should `topic_votes_yearly` contain Main Category tags only (29), Subcategory tags (~50–70), or all hierarchy levels (268)? Each has trade-offs:
   - Main Category only: Fewer rows, broader aggregation, risk of losing thematic detail
   - All levels: More rows, multi-tag inflation is higher, but full coverage
   - Recommended: Include both Main Category and Subcategory levels (not Specific Items), with a `TagLevel` column to distinguish them
3. **Fix the parser** to handle the flat tag format correctly, or restructure `un_votes_raw.tags` to use a JSON array with hierarchy metadata.

**Verification after fix**: 
```python
# After re-run, check tag count
tags_in_tv = df_topic_votes[df_topic_votes.Year == 2025]["TopicTag"].nunique()
assert tags_in_tv >= 29, f"Tag count regressed: {tags_in_tv}"
# Expect 50+ if Subcategories are included, 268 if all levels
```

---

### Issue 2 — 🔴 CRITICAL: 88.8% duplicate rows in `topic_votes_yearly`

**What we found**: 7,810 of 8,796 rows in `topic_votes_yearly` for 2025 are exact duplicates on `(Country, TopicTag)` with identical vote counts. After dedup, 4,891 rows remain.

**Root cause** (from source code review):

The `save_data_to_supabase()` function does `DELETE ... .neq('id', 0)` before inserting. If the pipeline was run twice, or if the delete didn't fully complete before the insert, duplicate rows would accumulate. Alternatively, if `generate_topic_votes()` produces duplicates internally (e.g., a tag appearing at multiple hierarchy levels resolves to the same string), the output DataFrame itself may contain dupes.

**Suggested fix**:
1. **Add deduplication before insert**: `df_topic_votes.drop_duplicates(subset=["Year", "Country", "TopicTag"], inplace=True)`
2. **Add a unique constraint** on `(Year, Country, TopicTag)` in the Supabase table schema to prevent future duplicates.
3. **Verify the delete-then-insert is atomic** — wrap in a transaction or use upsert instead.

**Verification after fix**:
```python
dupes = df_tv.duplicated(subset=["Country", "TopicTag"], keep=False)
assert dupes.sum() == 0, f"{dupes.sum()} duplicate rows found"
```

---

### Issue 3 — 🟠 HIGH: `topic_votes_yearly` undercounts (0.3–0.8× annual totals)

**What we found**: Summing `TotalVotes_Topic` across all tags for a country gives totals **lower** than `annual_scores.Total Votes in Year`. With multi-tagging, the sum should be **higher** (each resolution maps to multiple tags).

| Country | Annual Total | Topic Sum (deduped) | Ratio |
|---------|:-----------:|:-------------------:|:-----:|
| USA | 192 | 83 | 0.43× |
| BRA | 192 | 83 | 0.43× |
| ARG | 155 | 50 | 0.32× |

**Root cause**: Direct consequence of Issue #1. With only 29 of 268 tags surviving, many resolutions are not covered by any tag in the output. The 29 surviving tags collectively cover only ~83 of the 192 resolutions for a fully-participating country.

**Fix**: Resolving Issue #1 will fix this. After re-run, verify:
```python
# Topic sum should be >= annual total for fully-participating countries
for iso in ["USA", "GBR", "BRA"]:
    topic_sum = df_tv[df_tv.Country == iso]["TotalVotes_Topic"].sum()
    annual_total = df_ann[df_ann["Country name"] == iso]["Total Votes in Year"].iloc[0]
    assert topic_sum >= annual_total, f"{iso}: topic_sum={topic_sum} < annual={annual_total}"
```

---

### Issue 4 — 🟠 HIGH: Pairwise similarity stale / miscomputed for 2025

**What we found**: Replicating the **exact pipeline logic** (`generate_similarity_matrix()`) from `un_votes_with_sc` (192 GA resolutions, vote encoding YES=1/NO=-1/ABSTAIN=0/null=0, `sklearn.metrics.pairwise.cosine_similarity`) produces values that **do not match** the stored `pairwise_similarity_yearly` table.

| Metric | Value |
|--------|-------|
| Mean absolute difference | **0.151** |
| Pairs with diff > 0.1 | **10,335 / 18,528 (56%)** |
| Pairs matching within 0.001 | 485 (2.6%) |

**Specific examples**:

| Pair | Recomputed | Stored | Diff |
|------|:----------:|:------:|:----:|
| GBR–USA | -0.478 | -0.694 | 0.216 |
| ISR–USA | +0.658 | +0.462 | 0.196 |
| CHN–RUS | +0.784 | +0.886 | 0.102 |
| DEU–FRA | +0.938 | +1.000 | 0.062 |

**762 pairs stored as 0.0** when they should have non-zero similarity:
- 383 involve AFG/VEN (correct — zero votes)
- **379 are spurious**, including:
  - BOL: 190 zero-sim pairs (essentially all pairs), despite having 40 votes
  - ARG: 56 zero-sim pairs, despite 155 votes
  - SWZ: 56 zero-sim pairs, despite 84 votes

**Root cause**: The stored data was computed from an **older/incomplete snapshot** of the source data. BOL being treated as a non-voter (all 190 pairs = 0) confirms the pipeline ran before BOL's 2025 votes were added to `un_votes_with_sc`. The pipeline was not re-run after the source data was updated.

**Suggested fix**: Simply **re-run `generate_similarity_matrix()`** against the current `un_votes_with_sc` data. No code changes needed — the logic is correct, the input data was just stale.

**Verification after fix**:
```python
# Spot-check 5 pairs by recomputing from raw
# See analysis/4C/validate_pairwise_pipeline.py for the full verification script
# Mean diff should be < 0.0001
```

---

### Issue 5 — 🟠 HIGH: AFG/VEN inconsistent across tables

**What we found**: Afghanistan and Venezuela have zero votes in 2025 (all null in `un_votes_with_sc`). They appear in `pairwise_similarity_yearly` (with 0.0 similarity to everyone) but are **absent** from `annual_scores` (191 countries instead of 193).

**Root cause** (from source code): The `generate_combined_index()` function counts votes by melting the wide-format data. Countries with all-null votes will have zero rows after filtering to `YES/NO/ABSTAIN`, so they never get a row in the vote counts DataFrame, and therefore never get a row in `annual_scores`. But `generate_similarity_matrix()` includes them because it uses `fillna(0)` — a zero vector maps to 0 similarity with everyone.

**Suggested fix**: Choose one consistent policy:
- **Option A (recommended)**: Exclude non-voting countries from all tables. Add a filter in `generate_similarity_matrix()`:
  ```python
  # After vote encoding, remove countries with all-zero vectors
  active_cols = [c for c in country_cols if vote_matrix_numeric[c].any()]
  vote_matrix_numeric = vote_matrix_numeric[active_cols]
  ```
- **Option B**: Include them in all tables with null scores. Add empty rows for AFG/VEN in `annual_scores` with null pillar scores and 0 votes.

**Verification**:
```python
ann_countries = set(df_annual[df_annual.Year == 2025]["Country name"])
pw_countries = set(df_pw[df_pw.Year == 2025]["Country1_ISO3"]) | set(df_pw[df_pw.Year == 2025]["Country2_ISO3"])
assert ann_countries == pw_countries, f"Mismatch: {ann_countries ^ pw_countries}"
```

---

### Issue 6 — 🟠 HIGH: BOL missing from `topic_votes_yearly`

**What we found**: Bolivia has 40 votes in `annual_scores` for 2025 but zero rows in `topic_votes_yearly`.

**Root cause**: BOL also has 190 spurious zero-sim pairs in pairwise (see Issue #4), confirming BOL's votes were added to `un_votes_with_sc` **after** the pipeline last ran. The topic_votes table was generated from the same stale snapshot.

**Fix**: Re-running the pipeline against current data will include BOL. No code change needed.

**Verification**: `df_tv[(df_tv.Year == 2025) & (df_tv.Country == "BOL")]` should return rows.

---

### Issue 7 — 🟡 MEDIUM: `annual_scores` — "Pillar Score" columns contain normalized values, not raw

**What we found** (from source code review): The `generate_annual_scores()` function explicitly overwrites the raw pillar scores with their normalized counterparts:

```python
# In generate_annual_scores():
df_annual['Pillar 1 Score'] = df_annual['Pillar 1 Normalized']
df_annual['Pillar 2 Score'] = df_annual['Pillar 2 Normalized']
df_annual['Pillar 3 Score'] = df_annual['Pillar 3 Normalized']
df_annual['Total Index Average'] = df_annual['Total Index Normalized']
```

This means the `Pillar 1 Score` column in `annual_scores` is actually the **min-max normalized** value (0–100 per year), not the raw computation output. The raw scores are discarded.

**Impact**: The column name is misleading. When we say "P1 score dropped from 72.0 to 48.2", this is a normalized score, not an absolute metric. Year-over-year comparisons are valid (both are normalized within their year), but the absolute number has no intrinsic meaning beyond its rank position.

**Suggested fix**: Either:
1. Rename the column to `Pillar 1 Score (Normalized)` for clarity, or
2. Keep both raw and normalized in the table, or
3. Document this in the schema description

---

### Issue 8 — 🟡 MEDIUM: `save_data_to_supabase()` truncates CosineSimilarity to 4 decimal places

**What we found** (from source code review): The save function does:
```python
df_to_upload[col] = df_to_upload[col].apply(lambda x: round(x, 4) if ... else x)
```

Then converts to string:
```python
df_to_upload[col] = df_to_upload[col].astype(str)
```

**But** the stored values have more than 4 decimal places (e.g., `-0.6940220937885672`). This means either:
1. The current data was uploaded by a different version of the save function, or
2. The string conversion preserved full precision despite the round() call

**Impact**: Minimal — the precision is sufficient for analysis. But the code as written would truncate to 4 decimal places on next re-run, changing stored values.

**Suggested fix**: Remove the `round(x, 4)` for `CosineSimilarity` — cosine similarity benefits from full float precision. Or at minimum, use `round(x, 10)`.

Also: the `abs(x) > 1e3` filter will silently null-out any score > 1000. For `CosineSimilarity` (range [-1, 1]) this is fine, but for other numeric columns it could be destructive.

---

### Issue 9 — 🟡 MEDIUM: No minimum-vote threshold for P1 scores

**What we found**: Countries with very few votes get extreme P1 scores:

| Country | P1 Score | Total Votes 2025 |
|---------|:--------:|:----------------:|
| STP | 100.0 | 12 |
| MDG | 49.6 | 11 |
| VCT | 70.9 | 16 |

**Root cause** (from source code): The `calculate_alignment_score_p1()` function has no minimum vote count. A country voting on 12 resolutions can produce P1=100 if all votes happen to be in the same direction for each tag.

**Suggested fix**: Add either:
1. A minimum total-vote threshold (e.g., `if total_votes < 50: return np.nan`)
2. A confidence/reliability column based on sample size

---

### Issue 10 — 🟡 MEDIUM: P1 uses a 4-year rolling window (`bloc_size=4`)

**What we found** (from source code): P1 (Internal Alignment) is not computed from a single year's votes. It uses a **4-year rolling bloc** (`bloc_size_p1=4`). For 2025, the bloc is [2022, 2023, 2024, 2025].

**Impact on analysis**: 
- P1 changes from 2024→2025 reflect **not just** 2025 voting behavior, but the fact that 2021 dropped out of the window and 2025 entered.
- A country's P1 can change even if its 2025 voting is identical to 2024, because 2021 votes (often unusual due to COVID/Ukraine) exit the window.
- The 2025 resolution count (~192) is roughly double 2024 (~95), so the 2025 year has disproportionate weight in the bloc.

**This is by design**, not a bug. But downstream reports should note that P1 is a 4-year rolling metric, not a single-year snapshot.

---

### Issue 11 — 🟡 MEDIUM: P1 formula floor effect (ARG = 0.0 on 155 votes)

**What we found**: Argentina has P1=0.0 in 2025 despite casting 155 votes (Y=42, N=84, A=29). It's the only country with P1=0 on a non-trivial vote count.

**Root cause** (from source code): The formula is:
```python
score = max(0.0, 1.0 - (sum(weighted_deviations) / total_votes_all_consistent_tags))
```

The `max(0.0, ...)` clamp means any country whose weighted deviations exceed total_votes will get exactly 0.0. Argentina's votes are highly inconsistent within topics across the 4-year window (2022–2025), especially because the Milei government reversed many voting positions in 2024–2025. The formula doesn't distinguish between "barely zero" and "deeply negative" — both map to 0.0.

**Suggested fix**: Consider reporting the unclamped value alongside the clamped one, so analysts can see the magnitude of inconsistency:
```python
raw_score = 1.0 - (sum(weighted_deviations) / total_votes_all_consistent_tags)
clamped_score = max(0.0, raw_score)
# Store both: Pillar1_Raw and Pillar1 (clamped)
```

---

## Part 4: Recomputation Verification Checklist

After re-running the pipeline, run these checks:

### `annual_scores`
- [ ] 191 countries for 2025 (or 193 if including AFG/VEN)
- [ ] `Yes + No + Abstain == Total` for every row
- [ ] P1 scores in [0, 100], no nulls (except possibly low-vote countries if threshold added)
- [ ] P1 Rank ordering consistent with P1 Score ordering
- [ ] Max `Total Votes in Year` = 192

### `topic_votes_yearly`
- [ ] No duplicate `(Year, Country, TopicTag)` rows
- [ ] Unique tags > 29 (should be 50+ with Subcategories, or 268 with all levels)
- [ ] `YesVotes + NoVotes + AbstainVotes == TotalVotes` for every row
- [ ] All 191 countries present (or 190 if BOL excluded by threshold)
- [ ] No junk tags (`test`, `data-type-fix`, empty strings)

### `pairwise_similarity_yearly`
- [ ] 18,528 pairs for 2025 (if 193 countries) or 18,145 (if 191)
- [ ] All similarities in [-1, 1]
- [ ] No self-pairs
- [ ] Non-voting countries excluded (AFG/VEN) or consistently handled
- [ ] Spot-check 5+ pairs match recomputation from `un_votes_with_sc` within 0.0001

### Cross-table consistency
- [ ] Same country set in all three tables
- [ ] Same year range in all three tables

### Existing verification scripts
Run these from the repo root (they validate the CSV extracts but the checks apply to DB data too):
```bash
python validation/validate_annual_scores.py
python validation/validate_pairwise_similarity.py
python validation/validate_topic_votes.py
python validation/validate_cross_files.py
python validation/validate_2025.py
```

---

## Part 5: Architecture Recommendations

These are not blocking the re-run but would prevent future issues:

### 5.1 Add unique constraints to Supabase tables

```sql
ALTER TABLE topic_votes_yearly
  ADD CONSTRAINT uq_topic_votes UNIQUE ("Year", "Country", "TopicTag");

ALTER TABLE pairwise_similarity_yearly
  ADD CONSTRAINT uq_pairwise UNIQUE ("Year", "Country1_ISO3", "Country2_ISO3");
```

### 5.2 Use upsert instead of delete-then-insert

The current `save_data_to_supabase()` deletes all rows then re-inserts. This is non-atomic and can leave the table empty if the insert fails. Use Supabase's `upsert()` instead:
```python
supabase.table(table_name).upsert(batch_rows, on_conflict="Year,Country,TopicTag").execute()
```

### 5.3 Add a pipeline metadata table

Track when each table was last computed, from which source data, and with what parameters:
```sql
CREATE TABLE pipeline_runs (
  id serial PRIMARY KEY,
  table_name text NOT NULL,
  source_table text NOT NULL,
  source_row_count int,
  output_row_count int,
  year_range text,
  pipeline_version text,
  started_at timestamp DEFAULT now(),
  completed_at timestamp,
  status text -- 'success', 'failed', 'partial'
);
```

### 5.4 Remove `round(x, 4)` and `abs(x) > 1e3` guard in save function

The CosineSimilarity rounding to 4 decimals loses precision. The `abs(x) > 1e3` filter silently nullifies any value over 1000, which is fine for similarity but could be dangerous for other numeric columns. Replace with column-specific guards.

### 5.5 Document the P1 computation

Add a markdown file or docstring explaining:
- P1 uses a 4-year rolling window
- Within-topic consistency is measured as deviation from the bloc's average vote distribution
- Scores are clamped to [0, 100] and then min-max normalized per year
- The final `Pillar 1 Score` stored in `annual_scores` is the **normalized** value, not the raw score

---

## Appendix: Audit Scripts Reference

| Script | Purpose | Run from |
|--------|---------|----------|
| `analysis/4C/pipeline_audit.py` | Tag gap analysis, duplicate detection, country coverage | repo root |
| `analysis/4C/supabase_audit.py` | Full Supabase table health check | repo root |
| `analysis/4C/supabase_deep_check.py` | Vote count mismatch root cause, cosine encoding tests | repo root |
| `analysis/4C/validate_pairwise_pipeline.py` | Exact pipeline replication for pairwise validation | repo root |
| `analysis/4C/check_zero_pairs.py` | Zero-similarity pair analysis | repo root |

All scripts use the `.env` credentials and query Supabase directly. Run with:
```bash
.venv/bin/python analysis/4C/<script_name>.py
```
