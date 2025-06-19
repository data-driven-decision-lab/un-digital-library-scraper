import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import os
import argparse
import logging
import glob
import re
from tqdm import tqdm
import itertools

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---
def find_latest_input_csv(pipeline_output_dir):
    """Finds the most recent raw voting data CSV file."""
    pattern = os.path.join(pipeline_output_dir, 'UN_VOTING_DATA_RAW_WITH_TAGS_*.csv')
    list_of_files = glob.glob(pattern)
    if not list_of_files:
        logging.warning(f"No CSV files found in '{pipeline_output_dir}' matching the pattern.")
        return None
    try:
        latest_file = max(list_of_files, key=os.path.getmtime)
        logging.info(f"Found latest input file: {latest_file}")
        return latest_file
    except Exception as e:
        logging.error(f"Error determining latest file: {e}. Please specify the input file manually.")
        return None

def identify_country_columns(df_columns):
    """Identifies likely country ISO3 columns (3 uppercase letters)."""
    potential_countries = []
    # Basic check: 3 characters, all uppercase
    for col in df_columns:
        if isinstance(col, str) and len(col) == 3 and col.isupper():
            potential_countries.append(col)
    # Exclude known non-countries that might fit the pattern
    known_non_countries = {'YES', 'NO'} # Add others if necessary
    country_cols = sorted([col for col in potential_countries if col not in known_non_countries])
    logging.info(f"Identified {len(country_cols)} potential country columns.")
    return country_cols

def map_vote(vote):
    """Maps vote string to numerical value."""
    if pd.isna(vote): return 0
    vote_str = str(vote).upper().strip()
    if vote_str == 'YES': return 1
    if vote_str == 'NO': return -1
    if vote_str == 'ABSTAIN': return 0 # Or use another value if needed
    return 0 # Treat non-votes or other values as 0

# --- Main Logic ---
def calculate_similarity_yearly(input_csv_path, output_csv_path):
    """
    Calculates pairwise cosine similarity between countries yearly.

    Args:
        input_csv_path (str): Path to the raw voting data CSV
                              (e.g., UN_VOTING_DATA_RAW_WITH_TAGS_YYYY-MM-DD.csv).
        output_csv_path (str): Path to save the yearly similarity results CSV.
    """
    logging.info(f"Reading raw voting data from: {input_csv_path}")
    try:
        df_raw = pd.read_csv(input_csv_path, low_memory=False)
        logging.info(f"Loaded {len(df_raw)} rows.")
    except FileNotFoundError:
        logging.error(f"ERROR: Input file not found at {input_csv_path}")
        return
    except Exception as e:
        logging.error(f"ERROR: Failed to load input CSV: {e}")
        return

    # --- Filter out Security Council Resolutions ---
    if 'Resolution' in df_raw.columns:
        initial_count = len(df_raw)
        # Resolutions starting with 'S/' are from the Security Council.
        df_raw = df_raw[~df_raw['Resolution'].str.startswith('S/', na=False)].copy()
        logging.info(f"Filtered out Security Council resolutions. Kept {len(df_raw)} of {initial_count} records.")
    else:
        logging.warning("'Resolution' column not found. Cannot filter out Security Council votes.")

    # --- Data Preparation ---
    # Ensure 'Year' is numeric
    if 'Year' not in df_raw.columns:
        logging.warning("'Year' column not found, attempting to extract from 'Date'.")
        if 'Date' in df_raw.columns:
             try:
                df_raw['Year'] = pd.to_datetime(df_raw['Date'], errors='coerce').dt.year
             except Exception as date_err:
                 logging.error(f"Error parsing 'Date' column: {date_err}")
                 return
        else:
             logging.error("Cannot determine year. Missing 'Year' and 'Date' columns.")
             return
    else:
        df_raw['Year'] = pd.to_numeric(df_raw['Year'], errors='coerce')

    df_raw.dropna(subset=['Year'], inplace=True)
    df_raw['Year'] = df_raw['Year'].astype(int)
    # No need for Period column anymore
    # df_raw['Period'] = df_raw['Year'].apply(get_5yr_period_label)
    # df_raw.dropna(subset=['Period'], inplace=True)

    # Identify country columns
    country_cols = identify_country_columns(df_raw.columns)
    if not country_cols:
        logging.error("ERROR: No country columns identified. Cannot calculate similarity.")
        return

    # --- Similarity Calculation Loop ---
    all_year_similarities = []
    unique_years = sorted(df_raw['Year'].unique())

    logging.info(f"Calculating similarities for {len(unique_years)} years...")
    for year in tqdm(unique_years, desc="Calculating Similarity per Year"):
        df_year = df_raw[df_raw['Year'] == year].copy()
        if df_year.empty:
            logging.warning(f"Skipping empty year: {year}")
            continue

        logging.debug(f"Processing year {year} with {len(df_year)} resolutions.")

        # Select only country columns for this year
        vote_matrix = df_year[country_cols]

        # Map votes to numerical values
        vote_matrix_numeric = vote_matrix.apply(lambda col: col.map(map_vote)) # Use apply with map

        # Fill any remaining NaNs
        vote_matrix_numeric = vote_matrix_numeric.fillna(0).astype(np.int8)

        # Ensure matrix is not empty after potential filtering/cleaning
        if vote_matrix_numeric.empty or vote_matrix_numeric.shape[0] == 0:
            logging.warning(f"Vote matrix is empty for year {year} after processing. Skipping.")
            continue

        # Calculate cosine similarity
        try:
            similarity_matrix = cosine_similarity(vote_matrix_numeric.T)
            df_sim = pd.DataFrame(similarity_matrix, index=country_cols, columns=country_cols)

            # Melt the similarity matrix into a long format
            df_sim_long = df_sim.stack().reset_index()
            df_sim_long.columns = ['Country1_ISO3', 'Country2_ISO3', 'CosineSimilarity']

            # Add the year column
            df_sim_long['Year'] = year

            # Filter out self-comparisons and duplicates
            df_sim_long = df_sim_long[df_sim_long['Country1_ISO3'] < df_sim_long['Country2_ISO3']]

            all_year_similarities.append(df_sim_long)
            logging.debug(f"Finished similarity calculation for year {year}")

        except ValueError as ve:
             logging.error(f"ValueError during similarity calculation for year {year}: {ve}. Check data.")
             continue
        except Exception as e:
             logging.error(f"Unexpected error during similarity calculation for year {year}: {e}")
             continue

    # --- Combine and Save Output ---
    if not all_year_similarities:
        logging.warning("No similarity results generated.")
        return

    final_df = pd.concat(all_year_similarities, ignore_index=True)

    # Reorder columns
    final_df = final_df[['Year', 'Country1_ISO3', 'Country2_ISO3', 'CosineSimilarity']]

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    logging.info(f"Saving pairwise yearly similarity results to: {output_csv_path}")
    try:
        final_df.to_csv(output_csv_path, index=False, float_format='%.6f')
        logging.info(f"Successfully saved {len(final_df)} pairwise similarity rows.")
    except Exception as e:
        logging.error(f"ERROR: Failed to save output CSV: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Determine script's directory to build relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Calculate pairwise country voting similarity yearly.")
    # Keep -p argument for flexibility
    parser.add_argument("-p", "--pipeline-output-dir", default="../pipeline_output",
                        help="Directory containing the raw voting data CSVs (relative to script location, default: ../pipeline_output)")

    args = parser.parse_args()

    # Construct absolute path for pipeline_output_dir relative to script location
    pipeline_output_dir_abs = os.path.abspath(os.path.join(script_dir, args.pipeline_output_dir))

    # Determine input file path using the absolute path
    input_path = find_latest_input_csv(pipeline_output_dir_abs)

    # Determine fixed output path relative to script location
    output_filename = "pairwise_similarity_yearly.csv" # Changed output filename
    output_path = os.path.abspath(os.path.join(script_dir, "..", "un_report_api", "app", "required_csvs", output_filename))

    if not input_path:
        logging.error("No input file specified or found. Exiting.")
    else:
        calculate_similarity_yearly(input_path, output_path) # Changed function call
        logging.info("Script finished.") 