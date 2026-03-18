---
phase: 02-database-migration
plan: 03
subsystem: database
tags: [turso, libsql, libsql_experimental, scraper, pipeline, sqlite, json]

# Dependency graph
requires:
  - phase: 02-database-migration/02-01
    provides: "Turso schema (un_votes_raw, un_votes_with_sc, pipeline_runs) + TursoDataLoader"
provides:
  - "Turso-native scraper pipeline: reads existing links from Turso, writes new vote rows with INSERT OR IGNORE"
  - "Wide-format DataFrame -> vote_data JSON blob conversion on insert"
  - "sc_flag derivation (Resolution starts with S/) on un_votes_with_sc inserts"
  - "pipeline_runs logging replacing scraper_logs Supabase writes"
affects: [03-pipeline-refactor, 04-api-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "INSERT OR IGNORE on Link (UNIQUE) to skip duplicate resolutions without error"
    - "Wide-format DataFrame -> {iso3: vote} JSON blob conversion before DB insert"
    - "libsql_experimental deferred import inside get_turso_connection() (consistent with turso_client.py)"
    - "pipeline_runs for scraper lifecycle logging (start INSERT, update notes, finish UPDATE)"

key-files:
  created: []
  modified:
    - src/un_data_pipeline/scraper_pipeline.py

key-decisions:
  - "get_turso_connection() defined inline in scraper_pipeline.py (same pattern as turso_client.py but independent module)"
  - "update_scraper_log() writes JSON-encoded updates to pipeline_runs.notes column (schema has no individual update columns)"
  - "All Supabase references removed; no SUPABASE_KEY or create_client anywhere in scraper_pipeline.py"

patterns-established:
  - "Country ISO3 columns identified by: len==3 and isupper() and not in {'YES','NO'}"
  - "vote_data serialization: json.dumps({col: row[col] for col in country_cols if pd.notna(row[col])})"

requirements-completed: [DB-03]

# Metrics
duration: 3min
completed: 2026-03-18
---

# Phase 2 Plan 3: Scraper Pipeline Turso Migration Summary

**scraper_pipeline.py rewritten from Supabase SDK to libsql_experimental: INSERT OR IGNORE deduplication, vote_data JSON blobs, and pipeline_runs lifecycle logging**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-18T20:33:55Z
- **Completed:** 2026-03-18T20:37:02Z
- **Tasks:** 1 of 1
- **Files modified:** 1

## Accomplishments

- Removed `from supabase import create_client, Client` and all `get_supabase_client()` calls
- Replaced `get_links_from_supabase()` with `get_links_from_turso()` reading from Turso `un_votes_with_sc`
- Replaced `upload_to_supabase_raw()` with `upload_to_turso_raw()` using `INSERT OR IGNORE` and `vote_data` JSON blob conversion
- Replaced `upload_to_supabase_with_sc()` with `upload_to_turso_with_sc()` including `sc_flag` derivation
- Replaced `get_all_data_from_supabase()` with `get_all_data_from_turso()` that expands `vote_data` JSON back to wide-format
- Replaced `scraper_logs` Supabase writes with `pipeline_runs` Turso INSERT/UPDATE lifecycle tracking

## Task Commits

Each task was committed atomically:

1. **Task 1: Replace Supabase functions with Turso equivalents** - `bc6c5c7` (feat)

**Plan metadata:** (committed below)

## Files Created/Modified

- `src/un_data_pipeline/scraper_pipeline.py` - Full Supabase-to-Turso migration; all read/write and logging operations now use libsql_experimental

## Decisions Made

- `get_turso_connection()` defined inline in `scraper_pipeline.py` rather than importing from `turso_client.py` — keeps the pipeline self-contained without cross-module coupling
- `update_scraper_log()` writes a JSON blob to `pipeline_runs.notes` since the schema doesn't have individual columns for the fine-grained update fields the scraper tracks (years_processed, record counts); final counts are written to `rows_affected` on finish
- `finish_scraper_log()` reads `rows_affected` from `scraper_log_data['new_records_processed']` — provides a meaningful row count for the pipeline_runs record

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required. Turso credentials (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) must be present in `.env` at runtime, same as all other Turso-enabled modules.

## Next Phase Readiness

- scraper_pipeline.py is fully Turso-native; no Supabase SDK dependency remains
- All three migration targets complete (schema, API data loader, scraper pipeline)
- Ready for Phase 3 pipeline refactor

---
*Phase: 02-database-migration*
*Completed: 2026-03-18*
