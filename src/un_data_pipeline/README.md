# Dashboard Data Pipeline

This directory contains the dashboard scoring pipeline in
`src/un_data_pipeline/dashboard_data_pipeline.py`.

## Overview

The pipeline reads from and writes to Turso (LibSQL):

1. Loads source vote data from Turso (`un_votes_with_sc` by default).
2. Filters out Security Council resolutions (`Resolution` starting with `S/`).
3. Generates:
   - `annual_scores.csv`
   - `topic_votes_yearly.csv`
   - `pairwise_similarity_yearly.csv`
4. Saves outputs to `src/un_report_api/app/required_csvs/`.
5. Validates output year coverage (must include `2025`) and fails on missing coverage.
6. Writes the three outputs to the Turso tables of the same name and records the run in `pipeline_runs`.

It uses `libsql-experimental` when installed, otherwise the HTTP client in `turso_http.py`. Table definitions are in `db/schema.sql` and `docs/SCHEMA.md`. Before the 2026 migration the pipeline used Supabase, which is now offline (see `legacy/supabase/`).

## Required Environment Variables

- `TURSO_DATABASE_URL` (required)
- `TURSO_AUTH_TOKEN` (required)
- `PIPELINE_SOURCE_TABLE` (optional, defaults to `un_votes_with_sc`)

## Manual Run

From the project root:

```bash
python -m src.un_data_pipeline.dashboard_data_pipeline
```

The script logs row counts/pages loaded from Turso and fails with a non-zero exit if required year coverage checks fail.
