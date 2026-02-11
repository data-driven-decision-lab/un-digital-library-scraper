# Dashboard Data Pipeline

This directory contains the dashboard scoring pipeline in
`src/un_data_pipeline/dashboard_data_pipeline.py`.

## Overview

The pipeline is Supabase-native:

1. Loads source vote data from Supabase.
2. Filters out Security Council resolutions (`Resolution` starting with `S/`).
3. Generates:
   - `annual_scores.csv`
   - `topic_votes_yearly.csv`
   - `pairwise_similarity_yearly.csv`
4. Saves outputs to `src/un_report_api/app/required_csvs/`.
5. Validates output year coverage (must include `2025`) and fails on missing coverage.

## Required Environment Variables

- `SUPABASE_KEY` (required)
- `SUPABASE_URL` (optional, defaults to project URL in code)
- `PIPELINE_SOURCE_TABLE` (optional, defaults to `un_votes_with_sc`)

## Manual Run

From the project root:

```bash
python -m src.un_data_pipeline.dashboard_data_pipeline
```

The script logs row counts/pages loaded from Supabase and fails with a non-zero exit if required year coverage checks fail.
