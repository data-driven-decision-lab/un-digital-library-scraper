# Combined Dashboard Data Pipeline

This directory contains a single, unified Python script, `pipeline.py`, responsible for processing raw UN voting data and generating all the necessary CSV files required by the project's reporting API.

## Overview

The `pipeline.py` script consolidates the logic from four different scripts that were previously scattered across the repository:

1.  `index_analysis/combined_index_calculation/combined_index_script.py`
2.  `dashboard_pipeline/annual_pillar_breakdown.py`
3.  `dashboard_pipeline/aggregate_topic_votes.py`
4.  `dashboard_pipeline/calculate_similarity_yearly.py`

By unifying them, this script simplifies the data generation process into a single, reliable step.

## How it Works

The script performs the following actions in a sequential and logical order:

1.  **Finds Raw Data**: It automatically locates the most recent `UN_VOTING_DATA_RAW_WITH_TAGS_*.csv` file from the `pipeline_output` directory.
2.  **Filters Data**: It removes all votes related to Security Council resolutions (those starting with `S/`).
3.  **Generates Outputs**: It processes the filtered data to generate three key CSV files:
    *   `annual_scores.csv`: Contains the three Pillar Scores, the Total Index Average, and associated ranks for each country and year.
    *   `topic_votes_yearly.csv`: Contains an aggregation of country votes (`YES`, `NO`, `ABSTAIN`) for each identified topic.
    *   `pairwise_similarity_yearly.csv`: Contains the cosine similarity score between every pair of countries for each year.
4.  **Saves Files**: It saves these three files directly into the `un_report_api/app/required_csvs/` directory, ensuring the API always has access to the latest data.

## How to Run

To execute the entire data pipeline, navigate to the project's root directory in your terminal and run the following command:

```bash
python combined_dashboard_pipeline/pipeline.py
```

The script will log its progress to the console, indicating which step is currently running and confirming when the final CSV files have been successfully saved. 