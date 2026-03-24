---
status: resolved
trigger: "pipeline-vote-undercount — USA 2025 shows 33 total votes in annual_scores, expected 192"
created: 2026-03-25T00:00:00Z
updated: 2026-03-25T01:40:00Z
---

## Current Focus

hypothesis: CONFIRMED — mixed timezone-aware/naive Date strings cause pd.to_datetime to coerce tz-naive values to NaT, dropping 159 new 2025 rows
test: Applied fix (corrected regex to strip tz suffixes before parsing), verified against Turso
expecting: 192 2025 GA rows, USA: 174 NO + 10 YES + 8 ABSTAIN = 192 total
next_action: RESOLVED — fix applied and verified, commit code

## Symptoms

expected: USA 2025 should show 192 total votes (174 No, 10 Yes, 8 Abstain) in annual_scores CSV/table
actual: USA 2025 shows only 33 total votes (30 No, 2 Yes, 1 Abstain) in annual_scores CSV
errors: No errors — pipeline completes successfully, just wrong counts
reproduction: Run dashboard_data_pipeline.py and check USA,2025 row in annual_scores.csv
started: Discovered today after scraper added 180 new resolutions

## Eliminated

- hypothesis: _expand_vote_data failing silently on new rows (None vote_data)
  evidence: 193 country columns correctly identified after load; USA column present in df; expand works fine
  timestamp: 2026-03-25T01:15:00Z

- hypothesis: SC filter incorrectly removing 2025 GA rows
  evidence: SC filter by Resolution.startswith('S/') correctly identifies 2,743 SC rows; 7,369 GA rows remain; 192 are 2025 GA rows
  timestamp: 2026-03-25T01:20:00Z

- hypothesis: Date format prevents pd.to_datetime from parsing 2025 dates
  evidence: Old rows have "2025-03-04 00:00:00+00" (tz-aware +00), new rows have "2025-09-19 00:00:00" (tz-naive). Both are parseable in isolation, but when pd.to_datetime sees a MIXED Series (some tz-aware, some tz-naive), it coerces tz-naive entries to NaT.
  CONFIRMED as root cause — not eliminated

## Evidence

- timestamp: 2026-03-25T00:01:00Z
  checked: data/un_votes_with_sc_expanded.csv (local stale CSV)
  found: 10,112 rows total, 33 2025 GA rows with USA vote data (30 NO, 2 YES, 1 ABSTAIN)
  implication: Local CSV is stale — was saved before 180 new resolutions were added

- timestamp: 2026-03-25T00:02:00Z
  checked: Turso un_votes_with_sc table via HTTP client
  found: 10,112 total rows, 192 2025 GA rows (sc_flag=0), 44 2025 SC rows (sc_flag=1)
  implication: Turso HAS the correct data. Pipeline loads all 10,112 rows.

- timestamp: 2026-03-25T00:03:00Z
  checked: vote_data column for 5 sample 2025 GA rows
  found: All have valid JSON strings, 193 keys each, USA present with correct values
  implication: vote_data JSON is properly formed for 2025 rows; expand is NOT the issue

- timestamp: 2026-03-25T01:20:00Z
  checked: Full pipeline load + SC filter + date parse on 10,112 rows
  found: 7,369 GA rows, but only 33 survive date parse to Year=2025. 159 rows dropped by dropna(subset=['Date']). Sample failing dates: '2025-09-19 00:00:00', '2025-10-29 00:00:00' (NO timezone suffix). Working dates: '2025-03-04 00:00:00+00' (WITH +00 suffix).
  implication: Mixed tz-aware/naive Series causes pandas to coerce tz-naive values to NaT

- timestamp: 2026-03-25T01:25:00Z
  checked: Pandas behavior with mixed-tz Series
  found: pd.to_datetime(errors='coerce') converts tz-naive strings to NaT when the Series contains any tz-aware value. This is expected pandas behavior (documented future breaking change).
  implication: The fix is to normalize all dates to tz-naive BEFORE parsing, by stripping the timezone suffix.

- timestamp: 2026-03-25T01:38:00Z
  checked: Fix verification — applied correct regex r'[+-]\d{2}(?::\d{2})?$' to strip tz suffixes
  found: 2025 GA rows now correctly = 192; USA 2025 = 174 NO + 10 YES + 8 ABSTAIN = 192 total
  implication: Fix is correct and verified

## Resolution

root_cause: dashboard_data_pipeline.py line 898 — `pd.to_datetime(df_filtered['Date'], errors='coerce')` silently coerces 159 newly-scraped 2025 rows to NaT. Old rows in Turso have Date strings like "2025-03-04 00:00:00+00" (timezone-aware, stored by load_csv_to_turso.py). New rows from the scraper pipeline have Date strings like "2025-09-19 00:00:00" (timezone-naive, stored by scraper_pipeline.py's upload_to_turso_with_sc). When both exist in the same pandas Series, pd.to_datetime coerces the tz-naive entries to NaT in mixed-tz mode, and dropna then removes all 159 new rows.

fix: Normalize the Date column before parsing by stripping timezone suffixes with regex r'[+-]\d{2}(?::\d{2})?$', making all values tz-naive before pd.to_datetime. Applied to both parse locations: main() at line 906 and validate_source_year_coverage() at line 305.

verification: After fix, pipeline loads 192 2025 GA rows (up from 33). USA 2025 = 174 NO + 10 YES + 8 ABSTAIN = 192 total. Matches expected values exactly.

files_changed:
  - src/un_data_pipeline/dashboard_data_pipeline.py (lines 305-308 and 906-909)
