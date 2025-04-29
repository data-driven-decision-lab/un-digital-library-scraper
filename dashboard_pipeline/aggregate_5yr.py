import pandas as pd
import numpy as np
import os
import argparse
import logging
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def get_5yr_period_label(year):
    """Calculates the 5-year period label for a given year."""
    if pd.isna(year):
        return None
    try:
        year_int = int(year)
        period_start = (year_int // 5) * 5
        period_end = period_start + 4
        return f"{period_start}-{period_end}"
    except (ValueError, TypeError):
        logging.warning(f"Could not convert year {year} to integer for period calculation.")
        return None

# --- Main Logic ---
def aggregate_scores(input_csv_path, output_csv_path):
    """
    Reads yearly scores and aggregates them into 5-year blocks.

    Args:
        input_csv_path (str): Path to the input yearly scores CSV
                              (e.g., combined_index_scores.csv).
        output_csv_path (str): Path to save the aggregated 5-year results CSV.
    """
    logging.info(f"Reading yearly scores from: {input_csv_path}")
    try:
        df_yearly = pd.read_csv(input_csv_path, low_memory=False)
        logging.info(f"Loaded {len(df_yearly)} rows.")
    except FileNotFoundError:
        logging.error(f"ERROR: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        logging.error(f"ERROR: Failed to load input CSV: {e}")
        return

    # --- Data Preparation ---
    # Ensure 'Year' is numeric
    df_yearly['Year'] = pd.to_numeric(df_yearly['Year'], errors='coerce')
    df_yearly.dropna(subset=['Year'], inplace=True)
    df_yearly['Year'] = df_yearly['Year'].astype(int)

    # Create 5-year period label
    logging.info("Calculating 5-year period labels...")
    df_yearly['Period'] = df_yearly['Year'].apply(get_5yr_period_label)
    df_yearly.dropna(subset=['Period'], inplace=True) # Remove rows where period couldn't be determined

    # Identify columns for aggregation
    # Assuming original column names from combined_index_script.py
    score_rank_cols = [
        'Pillar 1 Score', 'Pillar 2 Score', 'Pillar 3 Score',
        'Total Index Average', 'Overall Rank', 'Overall Rank Rolling Avg (3y)',
        'Total Index Normalized', 'Pillar 1 Normalized', 'Pillar 1 Rank',
        'Pillar 2 Normalized', 'Pillar 2 Rank', 'Pillar 3 Normalized', 'Pillar 3 Rank'
    ]
    vote_cols = ['Yes Votes', 'No Votes', 'Abstain Votes', 'Total Votes in Year']

    # Filter out columns that don't exist in the dataframe
    score_rank_cols = [col for col in score_rank_cols if col in df_yearly.columns]
    vote_cols = [col for col in vote_cols if col in df_yearly.columns]

    if not score_rank_cols and not vote_cols:
        logging.error("ERROR: No relevant score/rank or vote columns found for aggregation.")
        return

    # Define aggregation dictionary
    agg_dict = {}
    final_column_names = {}

    for col in score_rank_cols:
        # Ensure column is numeric before trying mean
        df_yearly[col] = pd.to_numeric(df_yearly[col], errors='coerce')
        agg_dict[col] = 'mean'
        final_column_names[col] = f"Avg_{col.replace(' ', '_')}"

    for col in vote_cols:
        # Ensure column is numeric before trying sum
        df_yearly[col] = pd.to_numeric(df_yearly[col], errors='coerce')
        agg_dict[col] = 'sum'
        final_column_names[col] = f"Total_{col.replace(' ', '_')}"
        
    # Drop rows where essential aggregation columns became NaN after numeric conversion
    essential_cols = list(agg_dict.keys())
    df_yearly.dropna(subset=essential_cols, how='all', inplace=True)

    # --- Aggregation ---
    grouping_cols = ['Country name', 'Period']
    # Check if 'Country name' exists, otherwise use 'Country' if available
    if 'Country name' not in df_yearly.columns and 'Country' in df_yearly.columns:
         grouping_cols = ['Country', 'Period']
         logging.warning("Using 'Country' column for grouping as 'Country name' not found.")
    elif 'Country name' not in df_yearly.columns and 'Country' not in df_yearly.columns:
        logging.error("ERROR: Neither 'Country name' nor 'Country' column found for grouping.")
        return

    logging.info(f"Grouping by {grouping_cols} and aggregating...")
    # Use observed=True for stability with potential Categorical data
    df_aggregated = df_yearly.groupby(grouping_cols, observed=True).agg(agg_dict)

    # --- Final Formatting ---
    df_aggregated = df_aggregated.rename(columns=final_column_names)
    df_aggregated = df_aggregated.reset_index()

    # --- Save Output ---
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    logging.info(f"Saving aggregated 5-year results to: {output_csv_path}")
    try:
        df_aggregated.to_csv(output_csv_path, index=False, float_format='%.4f')
        logging.info(f"Successfully saved {len(df_aggregated)} rows.")
    except Exception as e:
        logging.error(f"ERROR: Failed to save output CSV: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Determine script's directory to build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the assumed relative path to the input file from combined_index_script.py
    # Adjust this path if the index script saves its output elsewhere
    default_input_relative_path = "../index_analysis/combined_index_calculation/combined_index_output/combined_index_scores.csv"
    
    parser = argparse.ArgumentParser(description="Aggregate yearly UN voting scores into 5-year blocks.")
    # Add an optional argument to override the default input path if needed
    parser.add_argument("-i", "--input-index-file", default=default_input_relative_path,
                        help=f"Relative path to the yearly scores CSV file (default: {default_input_relative_path})")

    args = parser.parse_args()

    # Construct absolute path for the input index file
    input_path = os.path.abspath(os.path.join(script_dir, args.input_index_file))

    # Determine fixed output path relative to script location
    output_filename = "five_year_aggregates.csv"
    output_path = os.path.abspath(os.path.join(script_dir, "output", output_filename))

    aggregate_scores(input_path, output_path)
    logging.info("Script finished.") 