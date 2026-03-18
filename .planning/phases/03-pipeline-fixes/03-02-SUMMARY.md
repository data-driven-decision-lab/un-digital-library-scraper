---
phase: 03-pipeline-fixes
plan: 02
subsystem: pipeline
tags: [pandas, cosine-similarity, annual-scores, data-quality, precision, normalization]

# Dependency graph
requires:
  - phase: 03-pipeline-fixes
    plan: 01
    provides: tag-parser rewrite (subcategory_keys, parse_tags_for_subtag1 flat scan)
provides:
  - generate_similarity_matrix() filters zero-vote countries via active_cols before cosine computation
  - generate_annual_scores() drops rows where Total Votes in Year == 0 before returning
  - save_data_to_turso() stores all float score columns at full float64 precision, no rounding or overflow guard
  - generate_annual_scores() has inline comment documenting Pillar X Score = MIN-MAX normalized (0-100)
affects:
  - annual_scores table (AFG/VEN rows absent after pipeline re-run)
  - pairwise_similarity_yearly table (AFG/VEN pairs absent after pipeline re-run)
  - CosineSimilarity precision preserved end-to-end

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "active_cols filter: [c for c in vote_matrix_numeric.columns if vote_matrix_numeric[c].any()] after fillna(0)"
    - "Zero-vote row filter: df_annual[df_annual['Total Votes in Year'] > 0].copy() before return"
    - "Float None coercion: df.where(pd.notna(df), None) replaces apply(lambda) guard pattern"

key-files:
  created: []
  modified:
    - src/un_data_pipeline/dashboard_data_pipeline.py

key-decisions:
  - "Use active_cols (any() filter) rather than dropping rows — operates on the transposed vote matrix, not original df"
  - "Filter Total Votes in Year > 0 after numeric coercion (not before) so the column is guaranteed float for comparison"
  - "Replace abs(x) > 1e3 apply() + round() apply() with pd.where(pd.notna()) — simpler and vectorized"

patterns-established:
  - "df_sim uses active_cols (not country_cols) for both index and columns after PIPE-03 filter"

requirements-completed: [PIPE-03, PIPE-04, PIPE-05, PIPE-06]

# Metrics
duration: 15min
completed: 2026-03-19
---

# Phase 03 Plan 02: Zero-Vote Filter, Precision Fix, and Normalization Doc Summary

**Three targeted data quality fixes: zero-vote country exclusion from all output tables (PIPE-03), full float64 precision for CosineSimilarity (PIPE-04/05), and inline normalization documentation for Pillar Score columns (PIPE-06)**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-19T01:10:00Z
- **Completed:** 2026-03-19T01:25:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `active_cols` filter in `generate_similarity_matrix()`: after `fillna(0)`, countries whose entire vote vector is zero (all NaN votes mapped to 0) are excluded before cosine computation. Fixed companion bug: `df_sim` DataFrame now uses `active_cols` as index/columns instead of the stale `country_cols` list.
- Added `Total Votes in Year > 0` filter in `generate_annual_scores()`: rows with zero votes (AFG, VEN) are dropped before return, with a log entry counting dropped rows.
- Confirmed `generate_topic_votes()` requires no change — the existing `.isin(['YES','NO','ABSTAIN'])` filter already excludes null votes; verified by assertion.
- Removed `round(x, 4)` from `save_data_to_turso()` float column handling — LibSQL stores full float64 natively; CosineSimilarity values are now stored at full precision.
- Removed `abs(x) > 1e3` guard from `save_data_to_turso()` — was silently nullifying valid scores; not applicable to CosineSimilarity (range -1 to 1) or Pillar scores (range 0-100).
- Replaced both removed `apply(lambda)` calls with `df.where(pd.notna(df), None)` — vectorized and cleaner.
- Added PIPE-06 inline comment above the column overwrite block in `generate_annual_scores()` documenting that `Pillar X Score` stores the MIN-MAX normalized value (0-100), not the raw pillar computation output.

## Task Commits

Each task was committed atomically:

1. **Task 1: Exclude zero-vote countries from annual_scores and pairwise_similarity_yearly (PIPE-03)** - `b91176b` (fix)
2. **Task 2: Remove round(x,4) and abs(x)>1e3 guard; document Pillar Score normalization (PIPE-04/05/06)** - `714ae6b` (fix)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified

- `src/un_data_pipeline/dashboard_data_pipeline.py` - Four targeted edits: active_cols filter in generate_similarity_matrix(), df_sim index/columns corrected to active_cols, Total Votes in Year > 0 filter and PIPE-06 comment in generate_annual_scores(), simplified float handling in save_data_to_turso()

## Decisions Made

- `active_cols` computed from `vote_matrix_numeric.columns` (not `country_cols`) so the filter operates on the already-encoded matrix, not the raw string votes
- `df_sim` index/columns updated to `active_cols` — using the old `country_cols` would produce a shape mismatch (Rule 1 auto-fix applied)
- Filter placement in `generate_annual_scores()` is after numeric coercion so `Total Votes in Year` is guaranteed numeric for `> 0` comparison
- `pd.where(pd.notna(df), None)` is the idiomatic vectorized replacement for the two removed `apply(lambda)` calls

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] df_sim DataFrame constructor used stale country_cols after active_cols filter**
- **Found during:** Task 1 — first verification run
- **Issue:** After filtering `vote_matrix_numeric` to `active_cols`, the `pd.DataFrame(similarity_matrix, index=country_cols, columns=country_cols)` constructor still referenced the full `country_cols` list (e.g., 3 items), while `similarity_matrix` was (2x2) for 2 active countries. This caused a shape mismatch error: "Shape of passed values is (2, 2), indices imply (3, 3)".
- **Fix:** Changed `index=country_cols, columns=country_cols` to `index=active_cols, columns=active_cols` on the same line.
- **Files modified:** `src/un_data_pipeline/dashboard_data_pipeline.py`
- **Commit:** `b91176b` (included in Task 1 commit)

---

**Total deviations:** 1 (1 bug fix — direct consequence of the PIPE-03 filter introduced in Task 1)
**Impact on plan:** Fixed inline during verification; no additional commits required.

## Issues Encountered

- `libsql_experimental` not installable in test environment (known from Plan 01). Resolved with same MagicMock stub pattern.

## Self-Check

Files exist:
- `src/un_data_pipeline/dashboard_data_pipeline.py` - FOUND (modified)

Commits exist:
- `b91176b` - FOUND
- `714ae6b` - FOUND

## Self-Check: PASSED
