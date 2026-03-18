---
phase: 02-database-migration
plan: 02
subsystem: database
tags: [turso, libsql, libsql_experimental, pipeline, upsert, sqlite]

# Dependency graph
requires:
  - phase: 02-database-migration/02-01
    provides: Turso schema (annual_scores, topic_votes_yearly, pairwise_similarity_yearly, pipeline_runs, un_votes_with_sc), TursoDataLoader, get_turso_connection()

provides:
  - dashboard_data_pipeline.py reading source data from Turso un_votes_with_sc via libsql_experimental
  - Turso upsert writes to annual_scores, topic_votes_yearly, pairwise_similarity_yearly using INSERT OR REPLACE
  - pipeline_runs tracking: INSERT at start, UPDATE at end with status and rows_affected
  - No Supabase SDK dependency in pipeline code

affects: [03-api-migration, pipeline execution, data freshness]

# Tech tracking
tech-stack:
  added: [libsql_experimental (top-level import in pipeline), uuid, datetime]
  patterns: [INSERT OR REPLACE upsert pattern for idempotent writes, pipeline_runs lifecycle tracking (start/success/failure), column rename before DB write to match schema]

key-files:
  created: []
  modified:
    - src/un_data_pipeline/dashboard_data_pipeline.py

key-decisions:
  - "Column renames applied before DB write: Country name->Country for annual_scores; Country1_ISO3/Country2_ISO3->Country1/Country2 for pairwise_similarity_yearly — keeps processing code unchanged, adapts at persistence layer"
  - "libsql_experimental imported at module top-level in pipeline (unlike turso_client.py which defers it) — pipeline is always run with libsql installed, so eager import is cleaner"
  - "abs(x) > 1e3 guard retained for float columns (Phase 3 fix per PIPE-05) — removed only the string conversion step that was Supabase-specific"
  - "CSV files still written after Turso writes for API fallback compatibility"

patterns-established:
  - "Upsert pattern: INSERT OR REPLACE INTO table (cols) VALUES (?) with executemany in batches of 1000"
  - "pipeline_runs lifecycle: INSERT OR REPLACE at start with status=running, UPDATE at end with status=success/failed + rows_affected/error_message"
  - "Column mapping at persistence boundary: rename DataFrame columns to match schema before save, keep internal processing column names unchanged"

requirements-completed: [DB-03]

# Metrics
duration: 3min
completed: 2026-03-19
---

# Phase 02 Plan 02: Database Migration - Pipeline Turso Migration Summary

**Turso-native dashboard pipeline using libsql_experimental with INSERT OR REPLACE upserts and pipeline_runs lifecycle tracking replacing the Supabase SDK**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-18T20:33:06Z
- **Completed:** 2026-03-18T20:36:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Removed all Supabase SDK references from dashboard_data_pipeline.py (supabase import, SUPABASE_KEY, create_client, get_supabase_client, load_data_from_supabase, save_data_to_supabase)
- Added load_data_from_turso() that fetches all rows from Turso via libsql_experimental, with _expand_vote_data() to deserialise vote_data JSON blob into per-country columns
- Added save_data_to_turso() with INSERT OR REPLACE upsert logic, batched in groups of 1000, replacing the old delete-then-insert pattern
- Added pipeline_runs tracking with INSERT at pipeline start and UPDATE at completion (success or failure with error_message)
- Applied column renames at DB write boundary to match schema (Country name->Country, Country1_ISO3->Country1, Country2_ISO3->Country2)

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace Supabase client functions with Turso equivalents** - `b6a43f0` (feat)

**Plan metadata:** (docs commit — to follow)

## Files Created/Modified
- `src/un_data_pipeline/dashboard_data_pipeline.py` - Full Turso I/O replacement: get_turso_connection(), load_data_from_turso(), save_data_to_turso(), pipeline_runs tracking in main()

## Decisions Made
- Column renames applied at persistence layer boundary, not in processing functions — keeps generate_combined_index/generate_similarity_matrix outputs unchanged
- libsql_experimental imported at module top-level (not deferred like turso_client.py) since pipeline always runs with the package installed
- abs(x) > 1e3 guard kept for float columns per PIPE-05 backlog — only removed string-conversion step that was a Supabase JSON serialisation workaround
- CSV output preserved alongside Turso writes for API fallback path

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added column renaming before DB writes**
- **Found during:** Task 1 (save function implementation)
- **Issue:** Pipeline produces `Country name` column (annual_scores) and `Country1_ISO3`/`Country2_ISO3` columns (pairwise_similarity); Turso schema uses `Country`, `Country1`, `Country2` respectively
- **Fix:** Added explicit renames in main() before calling save_data_to_turso() — df_annual_scores_db and df_similarity_db with renamed columns
- **Files modified:** src/un_data_pipeline/dashboard_data_pipeline.py
- **Verification:** Column names match schema UNIQUE constraints (Year, Country) and (Year, Country1, Country2)
- **Committed in:** b6a43f0 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (missing critical — schema alignment)
**Impact on plan:** Required for correct INSERT OR REPLACE to hit the UNIQUE constraints. No scope creep.

## Issues Encountered
None — plan executed cleanly after identifying the column name mismatch between pipeline output and schema.

## User Setup Required
None - no external service configuration required. Turso credentials (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) were already noted as required in .env.

## Next Phase Readiness
- Pipeline is now Turso-native end-to-end: reads from un_votes_with_sc, writes to all three output tables, records pipeline_runs
- Phase 03 (API migration) can now use Turso as the source of truth for all tables
- No Supabase dependency remains in the pipeline layer

---
*Phase: 02-database-migration*
*Completed: 2026-03-19*
