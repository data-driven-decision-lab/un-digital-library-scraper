#!/usr/bin/env python3
"""Processes the combined_index_scores.csv file to prepare annual data for the dashboard."""

import pandas as pd
import numpy as np
import os
import argparse
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Logic ---
def process_annual_scores(input_csv_path, output_csv_path):
    """
    Reads yearly scores and saves selected/formatted annual data.

    Args:
        input_csv_path (str): Path to the input yearly scores CSV 
                              (e.g., combined_index_scores.csv).
        output_csv_path (str): Path to save the annual results CSV.
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
    
    # Identify core columns to keep (adjust as needed for dashboard)
    # Using original names from combined_index_script.py output
    country_col = 'Country name' if 'Country name' in df_yearly.columns else 'Country'
    if country_col not in df_yearly.columns:
        logging.error("ERROR: Neither 'Country name' nor 'Country' column found.")
        return
        
    core_cols = [
        country_col, 'Year',
        'Pillar 1 Score', 'Pillar 2 Score', 'Pillar 3 Score',
        'Total Index Average', 'Overall Rank', 'Overall Rank Rolling Avg (3y)',
        'Total Index Normalized', 'Pillar 1 Normalized', 'Pillar 1 Rank',
        'Pillar 2 Normalized', 'Pillar 2 Rank', 'Pillar 3 Normalized', 'Pillar 3 Rank',
        'Yes Votes', 'No Votes', 'Abstain Votes', 'Total Votes in Year'
    ]

    # Filter to only columns that exist in the input dataframe
    cols_to_keep = [col for col in core_cols if col in df_yearly.columns]
    
    if len(cols_to_keep) <= 2: # Only country and year found
        logging.error("ERROR: No relevant score/rank or vote columns found in the input file.")
        return
        
    df_annual = df_yearly[cols_to_keep].copy()
    logging.info(f"Selected {len(df_annual.columns)} columns for annual output.")

    # Optional: Rename columns for consistency (e.g., replace spaces with underscores)
    # df_annual.columns = [col.replace(' ', '_') for col in df_annual.columns]
    # logging.info("Renamed columns to use underscores.")

    # Ensure numeric types for score/vote columns (important for dashboard tools)
    numeric_cols = [col for col in cols_to_keep if col not in [country_col, 'Year']]
    for col in numeric_cols:
        df_annual[col] = pd.to_numeric(df_annual[col], errors='coerce')
        # Consider how NaNs should be handled - leave as NaN or fill?
        # df_annual[col] = df_annual[col].fillna(0) # Example: fill with 0

    # --- Save Output ---
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    logging.info(f"Saving annual results to: {output_csv_path}")
    try:
        df_annual.to_csv(output_csv_path, index=False, float_format='%.4f')
        logging.info(f"Successfully saved {len(df_annual)} rows.")
    except Exception as e:
        logging.error(f"ERROR: Failed to save output CSV: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Determine script's directory to build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the assumed relative path to the input file from combined_index_script.py
    # Adjust this path if the index script saves its output elsewhere
    default_input_relative_path = "../index_analysis/combined_index_calculation/combined_index_output/combined_index_scores.csv"
    
    parser = argparse.ArgumentParser(description="Process yearly UN voting scores for dashboard.")
    # Add an optional argument to override the default input path if needed
    parser.add_argument("-i", "--input-index-file", default=default_input_relative_path,
                        help=f"Relative path to the yearly scores CSV file (default: {default_input_relative_path})")

    args = parser.parse_args()

    # Construct absolute path for the input index file
    input_path = os.path.abspath(os.path.join(script_dir, args.input_index_file))

    # Determine fixed output path relative to script location
    output_filename = "annual_scores.csv" # Changed output filename
    output_path = os.path.abspath(os.path.join(script_dir, "output", output_filename))

    process_annual_scores(input_path, output_path) # Changed function call
    logging.info("Script finished.")
