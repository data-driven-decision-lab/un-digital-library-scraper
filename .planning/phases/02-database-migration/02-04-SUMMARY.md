---
phase: 02-database-migration
plan: 04
subsystem: api
tags: [turso, libsql, supabase-removal, imports, requirements]

requires:
  - phase: 02-database-migration/02-01
    provides: turso_client.py with turso_loader global instance matching SupabaseDataLoader interface

provides:
  - report_generator.py reads data via turso_loader (not supabase_loader)
  - ranking_generator.py reads data via turso_loader (not supabase_loader)
  - main.py free of supabase SDK imports
  - supabase_client.py deprecation stub raising ImportError on any residual import
  - requirements.txt updated with libsql-experimental, supabase package removed

affects: [03-api-testing, 04-cleanup]

tech-stack:
  added: [libsql-experimental>=0.0.5]
  patterns: [deprecation-stub-pattern for soft module retirement]

key-files:
  created: []
  modified:
    - src/un_report_api/app/report_generator.py
    - src/un_report_api/app/ranking_generator.py
    - src/un_report_api/app/main.py
    - src/un_report_api/app/supabase_client.py
    - requirements.txt

key-decisions:
  - "supabase_client.py retained as ImportError stub (not deleted) — serves as safety net catching any missed import; deletion deferred to Phase 4 cleanup"
  - "libsql-experimental added to root requirements.txt only — no separate pipeline requirements.txt exists"

patterns-established:
  - "Deprecation stub pattern: replace deprecated module with raise ImportError() body to surface missed references at startup rather than silently using wrong loader"

requirements-completed: [DB-04]

duration: 2min
completed: 2026-03-18
---

# Phase 2 Plan 4: API Layer Supabase-to-Turso Migration Summary

**Replaced all supabase_loader imports with turso_loader across API modules, removed supabase SDK from requirements, and installed an ImportError deprecation stub in supabase_client.py**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-18T20:39:58Z
- **Completed:** 2026-03-18T20:41:50Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- report_generator.py and ranking_generator.py now import `turso_loader` from `turso_client` — no supabase_loader references remain
- main.py Supabase SDK try/except import block removed; UNClassificationMapper guard retained as-is
- requirements.txt: `supabase>=2.0.0` removed, `libsql-experimental>=0.0.5` added
- supabase_client.py replaced with a deprecation stub that raises `ImportError` immediately on import — any residual import elsewhere surfaces at startup rather than silently loading the wrong data source

## Task Commits

Each task was committed atomically:

1. **Task 1: Update API module imports from supabase_loader to turso_loader** - `eedde94` (feat)
2. **Task 2: Remove Supabase package, add libsql-experimental, deprecate supabase_client.py** - `9cb91ad` (feat)

## Files Created/Modified

- `src/un_report_api/app/report_generator.py` - Import changed line 21; 4 call sites renamed supabase_loader -> turso_loader
- `src/un_report_api/app/ranking_generator.py` - Import changed line 8; 2 call sites renamed supabase_loader -> turso_loader
- `src/un_report_api/app/main.py` - Removed `from supabase import create_client` from try/except block
- `src/un_report_api/app/supabase_client.py` - Replaced 159-line implementation with 12-line ImportError deprecation stub
- `requirements.txt` - Replaced `supabase>=2.0.0` with `libsql-experimental>=0.0.5`

## Decisions Made

- `supabase_client.py` retained as a stub (not deleted outright): raises `ImportError` on import so any lingering `from supabase_client import` elsewhere is caught immediately at API startup. Deletion deferred to Phase 4 cleanup pass.
- `libsql-experimental` added only to the root `requirements.txt` — no separate `src/un_data_pipeline/requirements.txt` exists in the repo.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The grep check for remaining supabase references surfaced a comment line in turso_client.py (`# Global instance — drop-in replacement for supabase_loader`) which is not a live import; no code change needed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The full API layer now reads exclusively via `turso_loader` from `turso_client.py`
- The deprecation stub in `supabase_client.py` will surface any future accidental re-introduction of supabase_client imports at startup
- Phase 3 (API testing) can proceed — import chain is clean and syntax-verified

---
*Phase: 02-database-migration*
*Completed: 2026-03-18*
