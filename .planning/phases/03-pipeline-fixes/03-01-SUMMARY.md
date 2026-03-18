---
phase: 03-pipeline-fixes
plan: 01
subsystem: pipeline
tags: [pandas, un_classification, topic-votes, tag-parser, deduplication]

# Dependency graph
requires:
  - phase: 02-database-migration
    provides: Turso-native dashboard_data_pipeline.py with generate_topic_votes()
provides:
  - subcategory_keys set at module top-level (136 subcategory entries)
  - parse_tags_for_subtag1() that matches both Main Category and Subcategory level tags
  - drop_duplicates on (Year, Country, TopicTag) in generate_topic_votes() output
affects:
  - topic_votes_yearly table coverage (29 -> 50+ unique TopicTags)
  - 03-pipeline-fixes further plans referencing tag coverage

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "subcategory_keys built at module top-level alongside main_category_keys by iterating un_classification.values()"
    - "Flat linear scan for tag matching instead of hierarchical while-loop walk"
    - "dict.fromkeys() for deduplication preserving insertion order"

key-files:
  created: []
  modified:
    - src/un_data_pipeline/dashboard_data_pipeline.py

key-decisions:
  - "Match tags at both Main Category and Subcategory levels — all items in un_classification key sets are valid UNBIS tags"
  - "Return [] (empty list) on no match rather than ['No Tag'] sentinel — callers use explode+dropna, sentinel created spurious rows"
  - "subcategory_keys built eagerly at import time rather than inside generate_topic_votes() — avoids recomputing per call"

patterns-established:
  - "Module-level key sets: build main_category_keys and subcategory_keys together in try/except ImportError block"
  - "No-sentinel policy: tag parsers return [] on miss, never a placeholder string"

requirements-completed: [PIPE-01, PIPE-02]

# Metrics
duration: 10min
completed: 2026-03-19
---

# Phase 03 Plan 01: Tag Parser & Dedup Fix Summary

**Flat linear tag scanner matching all 17 main categories + 136 subcategories instead of 17-only, plus drop_duplicates guard on topic_votes_yearly output**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-19T01:00:00Z
- **Completed:** 2026-03-19T01:10:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Added `subcategory_keys` (136 entries) at module top-level, built from all inner dict keys of `un_classification`
- Replaced broken while-loop `parse_tags_for_subtag1()` with flat linear scan checking both `main_category_keys` and `subcategory_keys`
- Removed `"No Tag"` sentinel — unmatched tags return `[]` and are silently dropped via `dropna` downstream
- Added `df_final.drop_duplicates(subset=['Year','Country','TopicTag'], inplace=True)` before `generate_topic_votes()` return

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite parse_tags_for_subtag1() to capture all matching tags** - `a961912` (fix)

**Plan metadata:** _(docs commit follows)_

## Files Created/Modified
- `src/un_data_pipeline/dashboard_data_pipeline.py` - Three targeted edits: subcategory_keys at module level, new parse_tags_for_subtag1() body, drop_duplicates in generate_topic_votes()

## Decisions Made
- `subcategory_keys` built at module import time (not inside the function) to avoid recomputing on every call
- Empty list return on no match instead of sentinel string — the `explode` + `dropna` pattern already handles this correctly
- `dict.fromkeys()` for order-preserving deduplication within a single tag string

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] libsql_experimental not installable in test environment**
- **Found during:** Task 1 verification
- **Issue:** `import libsql_experimental as libsql` at module top-level prevented importing the module for tests; package requires Rust/Cargo toolchain to build, which is unavailable
- **Fix:** Ran verification with `unittest.mock` stub (`sys.modules['libsql_experimental'] = MagicMock()`) — all three PASS assertions validated correctly
- **Files modified:** None (test-only workaround, no production code change)
- **Verification:** All three PASS lines printed; secondary grep checks passed
- **Committed in:** Verification-only workaround, not committed

---

**Total deviations:** 1 (1 blocking — test environment limitation, no production code impact)
**Impact on plan:** Verification strategy adapted; production code is unchanged and correct.

## Issues Encountered
- `libsql_experimental` requires Rust/Cargo to build from source; test environment lacks Rust toolchain. Resolved by mocking the module during verification — this does not affect production pipeline execution where the package is pre-installed.

## Self-Check

Files exist:
- `src/un_data_pipeline/dashboard_data_pipeline.py` - FOUND (modified)

Commits exist:
- `a961912` - FOUND

## Self-Check: PASSED

## Next Phase Readiness
- `generate_topic_votes()` now produces materially more TopicTag rows covering both hierarchy levels
- `topic_votes_yearly` will have zero duplicate (Year, Country, TopicTag) rows after next pipeline run
- Ready for Phase 03 Plan 02 (remaining pipeline fixes)

---
*Phase: 03-pipeline-fixes*
*Completed: 2026-03-19*
