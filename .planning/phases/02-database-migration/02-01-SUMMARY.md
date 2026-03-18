---
phase: 02-database-migration
plan: "01"
subsystem: database
tags: [turso, libsql, sqlite, schema, ddl, python, pandas]

requires:
  - phase: 01-cleanup
    provides: cleaned project structure with supabase_client.py as reference interface

provides:
  - db/schema.sql — SQLite/LibSQL DDL for all six tables with UNIQUE constraints
  - src/un_report_api/app/turso_client.py — TursoDataLoader class replacing SupabaseDataLoader

affects:
  - 02-database-migration (plans 02+): schema exists in Turso for pipeline writes
  - 03-pipeline-refactor: TursoDataLoader available for import in pipeline modules
  - 04-api-migration: turso_loader global replaces supabase_loader in report/ranking generators

tech-stack:
  added: [libsql-experimental (deferred import), db/schema.sql]
  patterns:
    - "Deferred libsql_experimental import inside get_turso_connection() to avoid ImportError at module load time"
    - "JSON blob pattern for wide-format vote data (vote_data TEXT) avoiding 190+ country column enumeration"
    - "UNIQUE constraints on composite keys for upsert-safety (topic_votes_yearly, pairwise_similarity_yearly)"

key-files:
  created:
    - db/schema.sql
    - src/un_report_api/app/turso_client.py
  modified: []

key-decisions:
  - "vote_data stored as JSON TEXT blob to avoid enumerating 190+ country ISO3 columns in DDL"
  - "libsql_experimental import deferred inside get_turso_connection() so API starts without it installed"
  - "TursoDataLoader retains identical CSV-reading logic from SupabaseDataLoader — only class name and Supabase SDK dependency removed"
  - "pipeline_runs table added for DB-06 execution metadata tracking"

patterns-established:
  - "Deferred heavy imports: wrap optional/heavyweight imports inside the function that needs them"
  - "Schema-first: DDL file exists at db/schema.sql as single source of truth for table structure"

requirements-completed: [DB-01, DB-02, DB-05, DB-06]

duration: 2min
completed: 2026-03-19
---

# Phase 02 Plan 01: Turso Schema and Client Foundation Summary

**SQLite/LibSQL DDL schema with six tables plus UNIQUE constraints, and a TursoDataLoader Python class replacing SupabaseDataLoader with identical CSV-reading interface and deferred libsql_experimental connection helper**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-18T20:27:54Z
- **Completed:** 2026-03-18T20:29:56Z
- **Tasks:** 2
- **Files modified:** 2 (created)

## Accomplishments

- Created `db/schema.sql` with all six tables (un_votes_raw, un_votes_with_sc, annual_scores, topic_votes_yearly, pairwise_similarity_yearly, pipeline_runs) verified against in-memory SQLite
- UNIQUE constraints on `topic_votes_yearly(Year, Country, TopicTag)` and `pairwise_similarity_yearly(Year, Country1, Country2)` satisfying DB-05
- Created `src/un_report_api/app/turso_client.py` with `TursoDataLoader` class and `turso_loader` global — zero Supabase SDK dependency, identical public interface to `SupabaseDataLoader`

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Turso schema SQL file** - `bff3f18` (feat)
2. **Task 2: Create TursoDataLoader client module** - `686c871` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `db/schema.sql` - Full DDL for six tables with UNIQUE constraints, pipeline_runs metadata table, JSON blob pattern for vote_data
- `src/un_report_api/app/turso_client.py` - TursoDataLoader class with five load_* methods (CSV-based), get_turso_connection() helper with deferred libsql_experimental import, turso_loader global instance

## Decisions Made

- **JSON blob for vote_data:** The scraper produces wide-format rows with 190+ country ISO3 columns. Rather than enumerate all in DDL (fragile, hard to maintain), vote data is stored as a `vote_data TEXT` JSON column. Design is documented in schema comments.
- **Deferred libsql_experimental import:** `get_turso_connection()` defers `import libsql_experimental as libsql` inside the function body so the API can start and serve CSV data even if the package isn't installed (e.g. in CI or during local dev without the native binary).
- **Verbatim CSV logic copy:** The five `load_*` methods are copied verbatim from `supabase_client.py` — only the class name and Supabase SDK import were removed. This eliminates the risk of silent behavioral drift between the old and new clients.

## Deviations from Plan

None — plan executed exactly as written.

The UNIQUE constraint check during Task 1 verification revealed SQLite internally creates a `sqlite_sequence` table due to AUTOINCREMENT columns. The plan's verification command counted 6 tables but SQLite exposes 7 (including `sqlite_sequence`). This was handled by filtering internal tables (`WHERE name NOT LIKE 'sqlite_%'`) — not a deviation from the schema design.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required beyond the already-noted Turso credentials in `.env`.

## Next Phase Readiness

- `db/schema.sql` is ready to be applied to the live Turso database (plan 02-02 will handle the apply step)
- `turso_client.py` is importable and ready for consumer modules to switch imports in plan 02-04
- No blockers for subsequent plans in this phase

---
*Phase: 02-database-migration*
*Completed: 2026-03-19*
