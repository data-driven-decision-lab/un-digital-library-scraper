"""Generates yearly pillar rankings for all countries."""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Dict, Tuple, Optional

# Configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "required_csvs")
# Path to the dictionaries directory, which is at the project root level
DICTIONARIES_DIR = os.path.join(os.path.dirname(os.path.dirname(APP_DIR)), "dictionaries")
ANNUAL_SCORES_FILE = "annual_scores.csv"
CLASSIFICATIONS_FILE = "country_classifications_2023.csv"
DECIMAL_PLACES = 2

# Configure logging
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger
# If this module is run directly, and not imported by FastAPI, basic logging can be set up.
# However, FastAPI's logging configuration usually takes precedence when run as part of the API.
if not logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)


def standardize_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes DataFrame column names to lowercase and replaces spaces with underscores."""
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

def calculate_rankings_for_year(df_year: pd.DataFrame, pillar_col: str) -> pd.DataFrame:
    """Calculates rankings for a given pillar in a specific year.
    Higher scores get better ranks (rank 1 is best).
    Handles NaNs by assigning them a rank of NaN and keeping them at the bottom if sorted.
    """
    # Sort by score descending (higher is better), NaNs are put last by default
    df_sorted = df_year.sort_values(by=pillar_col, ascending=False, na_position='last')
    # Rank, 'min' method assigns the same rank to ties (e.g., 1, 2, 2, 4)
    df_sorted[f'{pillar_col}_rank'] = df_sorted[pillar_col].rank(method='min', ascending=False)
    # Convert ranks to integer where possible (NaNs will remain float)
    df_sorted[f'{pillar_col}_rank'] = df_sorted[f'{pillar_col}_rank'].astype('Int64') # Pandas nullable integer type
    return df_sorted

def get_rankings_for_pillar(
    df_current_year: pd.DataFrame,
    df_previous_year: Optional[pd.DataFrame],
    df_10_year_ago: Optional[pd.DataFrame],
    pillar_name: str, 
    country_col: str = 'country_code'
) -> List[Dict]:
    """
    Generates a list of ranking entries for a specific pillar, including rank and value changes.
    """
    logger.debug(f"Processing rankings for pillar: {pillar_name}")

    # Rank for the current year
    df_current_ranked = calculate_rankings_for_year(df_current_year.copy(), pillar_name)

    results = []
    for _, row in df_current_ranked.iterrows():
        entry = {
            "country_name": row['country_name'],
            "country_code": row['country_code'],
            "value": round(row[pillar_name], DECIMAL_PLACES) if pd.notna(row[pillar_name]) else None,
            "rank": row[f'{pillar_name}_rank'] if pd.notna(row[f'{pillar_name}_rank']) else None,
            "rank_change": None,
            "value_change": None,
            "rank_change_10_year": None,
            "value_change_10_year": None,
            "is_oecd": bool(row['is_oecd']) if pd.notna(row['is_oecd']) else False,
            "is_g20": bool(row['is_g20']) if pd.notna(row['is_g20']) else False,
            "is_top_50_gdp": bool(row['is_top_50_gdp']) if pd.notna(row['is_top_50_gdp']) else False,
            "is_bottom_50_gdp": bool(row['is_bottom_50_gdp']) if pd.notna(row['is_bottom_50_gdp']) else False,
            "is_top_50_population": bool(row['is_top_50_population']) if pd.notna(row['is_top_50_population']) else False,
            "is_bottom_50_population": bool(row['is_bottom_50_population']) if pd.notna(row['is_bottom_50_population']) else False,
        }

        if df_previous_year is not None and not df_previous_year.empty:
            country_previous_data = df_previous_year[df_previous_year[country_col] == row[country_col]]
            if not country_previous_data.empty:
                prev_value = country_previous_data.iloc[0][pillar_name]
                prev_rank = country_previous_data.iloc[0].get(f'{pillar_name}_rank') 

                if pd.notna(prev_value) and pd.notna(entry["value"]):
                    entry["value_change"] = round(entry["value"] - prev_value, DECIMAL_PLACES)
                
                if pd.notna(prev_rank) and pd.notna(entry["rank"]):
                    entry["rank_change"] = int(prev_rank - entry["rank"])
            else:
                logger.debug(f"No previous year data for country {row[country_col]} in pillar {pillar_name}")
        else:
            logger.debug(f"No previous year DataFrame or it's empty for pillar {pillar_name}")

        if df_10_year_ago is not None and not df_10_year_ago.empty:
            country_10_year_data = df_10_year_ago[df_10_year_ago[country_col] == row[country_col]]
            if not country_10_year_data.empty:
                val_10_ago = country_10_year_data.iloc[0][pillar_name]
                rank_10_ago = country_10_year_data.iloc[0].get(f'{pillar_name}_rank')

                if pd.notna(val_10_ago) and pd.notna(entry["value"]):
                    entry["value_change_10_year"] = round(entry["value"] - val_10_ago, DECIMAL_PLACES)
                
                if pd.notna(rank_10_ago) and pd.notna(entry["rank"]):
                    entry["rank_change_10_year"] = int(rank_10_ago - entry["rank"])
            else:
                logger.debug(f"No 10-year-ago data for country {row[country_col]} in pillar {pillar_name}")
        else:
            logger.debug(f"No 10-year-ago DataFrame for pillar {pillar_name}")

        results.append(entry)
    
    results.sort(key=lambda x: (x['rank'] is None, x['rank'], x['country_name']))

    return results


def generate_yearly_rankings(year: int) -> Tuple[Dict, Optional[str]]:
    """
    Generates the full set of pillar rankings for a given year.
    Returns a dictionary of rankings and an optional message.
    """
    logger.info(f"Generating yearly rankings for {year}")
    annual_scores_path = os.path.join(DATA_DIR, ANNUAL_SCORES_FILE)
    classifications_path = os.path.join(DATA_DIR, CLASSIFICATIONS_FILE)
    message = None

    try:
        # Load scores and standardize.
        df_all_scores_raw = pd.read_csv(annual_scores_path)
        df_scores = standardize_col_names(df_all_scores_raw.copy())
        # The 'country_name' column in this file actually contains the 3-letter codes.
        # We rename it to 'country_code' to use as a reliable key for our merge.
        df_scores.rename(columns={'country_name': 'country_code'}, inplace=True)
        # Drop any leftover index columns from previous CSV saves to avoid confusion.
        if 'unnamed:_0' in df_scores.columns:
            df_scores = df_scores.drop(columns=['unnamed:_0'])
        logger.debug(f"Successfully loaded and standardized {ANNUAL_SCORES_FILE}")
    except FileNotFoundError:
        logger.error(f"Required data file not found: {annual_scores_path}")
        raise FileNotFoundError(f"The data file '{ANNUAL_SCORES_FILE}' was not found in '{DATA_DIR}'.")
    except Exception as e:
        logger.error(f"Error loading or processing {ANNUAL_SCORES_FILE}: {e}")
        raise ValueError(f"Error processing data file: {e}")

    try:
        df_classifications_raw = pd.read_csv(classifications_path)
        df_classifications = standardize_col_names(df_classifications_raw.copy())
        logger.debug(f"Successfully loaded and standardized {CLASSIFICATIONS_FILE}")
    except FileNotFoundError:
        logger.error(f"Required classification file not found: {classifications_path}")
        raise FileNotFoundError(f"The classification file '{CLASSIFICATIONS_FILE}' was not found.")
    
    # Select only the columns we need from the classification file to avoid merge conflicts.
    class_cols_to_merge = [
        'country_code', 'country_name', 'is_oecd', 'is_g20', 'is_top_50_gdp', 
        'is_bottom_50_gdp', 'is_top_50_population', 'is_bottom_50_population'
    ]
    df_class_subset = df_classifications[class_cols_to_merge]

    # Merge the scores data with the classification data.
    df_all_scores = pd.merge(df_scores, df_class_subset, on='country_code', how='left')

    # If a country from the scores file was not in the classification file,
    # its name will be missing (NaN). We fill it with the country code as a fallback.
    df_all_scores['country_name'].fillna(df_all_scores['country_code'], inplace=True)

    # Validate required columns after the merge
    required_cols = ['year', 'country_name', 'country_code', 'pillar_1_score', 'pillar_2_score', 'pillar_3_score', 'total_index_average']
    missing_cols = [col for col in required_cols if col not in df_all_scores.columns]
    if missing_cols:
        logger.error(f"Missing required columns in {ANNUAL_SCORES_FILE}: {missing_cols}")
        raise ValueError(f"Data file is missing required columns: {', '.join(missing_cols)}")

    # Ensure year column is integer
    df_all_scores['year'] = df_all_scores['year'].astype(int)
    
    # Filter data for the current year
    df_current_year = df_all_scores[df_all_scores['year'] == year].copy()
    if df_current_year.empty:
        logger.warning(f"No data found for the year {year} in {ANNUAL_SCORES_FILE}.")
        empty_rankings = {"year": year, "pillar_1_rankings": [], "pillar_2_rankings": [], "pillar_3_rankings": [], "average_pillar_rankings": []}
        return empty_rankings, f"No data available for the year {year}."

    # Filter data for the previous year (for change calculations)
    previous_year = year - 1
    df_previous_year_raw = df_all_scores[df_all_scores['year'] == previous_year].copy()

    if df_previous_year_raw.empty:
        logger.warning(f"No data found for the previous year ({previous_year}) for change calculations.")
        message = f"Data for the previous year ({previous_year}) is not available; rank and value changes cannot be calculated."
        df_previous_year_ranked = None
    else:
        # Pre-calculate ranks for all pillars for the previous year to make it comparable
        df_prev_p1_ranked = calculate_rankings_for_year(df_previous_year_raw.copy(), 'pillar_1_score')
        df_prev_p2_ranked = calculate_rankings_for_year(df_previous_year_raw.copy(), 'pillar_2_score')
        df_prev_p3_ranked = calculate_rankings_for_year(df_previous_year_raw.copy(), 'pillar_3_score')
        df_prev_avg_ranked = calculate_rankings_for_year(df_previous_year_raw.copy(), 'total_index_average')
        
        # Combine them by merging back on country_name and year (or just index if unique per year)
        # Assuming 'country_name' is unique per year
        df_previous_year_ranked = df_prev_p1_ranked.merge(
            df_prev_p2_ranked[['country_code', 'pillar_2_score_rank']], on='country_code', how='left'
        ).merge(
            df_prev_p3_ranked[['country_code', 'pillar_3_score_rank']], on='country_code', how='left'
        ).merge(
            df_prev_avg_ranked[['country_code', 'total_index_average_rank']], on='country_code', how='left'
        )
        logger.debug(f"Successfully processed data for previous year {previous_year}.")

    # Filter data for 10 years ago (for 10-year change calculations)
    year_10_ago = year - 10
    df_10_year_ago_raw = df_all_scores[df_all_scores['year'] == year_10_ago].copy()

    if df_10_year_ago_raw.empty:
        logger.warning(f"No data found for 10 years ago ({year_10_ago}) for change calculations.")
        message = (message or "") + f" Data for 10 years ago ({year_10_ago}) is not available; 10-year changes cannot be calculated."
        df_10_year_ago_ranked = None
    else:
        # Pre-calculate ranks for all pillars for 10 years ago to make it comparable
        df_10_p1_ranked = calculate_rankings_for_year(df_10_year_ago_raw.copy(), 'pillar_1_score')
        df_10_p2_ranked = calculate_rankings_for_year(df_10_year_ago_raw.copy(), 'pillar_2_score')
        df_10_p3_ranked = calculate_rankings_for_year(df_10_year_ago_raw.copy(), 'pillar_3_score')
        df_10_avg_ranked = calculate_rankings_for_year(df_10_year_ago_raw.copy(), 'total_index_average')
        
        # Combine them by merging back on country_name and year (or just index if unique per year)
        # Assuming 'country_name' is unique per year
        df_10_year_ago_ranked = df_10_p1_ranked.merge(
            df_10_p2_ranked[['country_code', 'pillar_2_score_rank']], on='country_code', how='left'
        ).merge(
            df_10_p3_ranked[['country_code', 'pillar_3_score_rank']], on='country_code', how='left'
        ).merge(
            df_10_avg_ranked[['country_code', 'total_index_average_rank']], on='country_code', how='left'
        )
        logger.debug(f"Successfully processed data for 10-year-ago {year_10_ago}.")

    # Pillar columns to iterate over
    pillar_mapping = {
        "pillar_1_rankings": "pillar_1_score",
        "pillar_2_rankings": "pillar_2_score",
        "pillar_3_rankings": "pillar_3_score",
        "average_pillar_rankings": "total_index_average" # Assuming this is the "Pillar Average"
    }

    all_rankings = {"year": year}

    for key_name, pillar_col_name in pillar_mapping.items():
        logger.info(f"Calculating rankings for: {key_name} (using column {pillar_col_name})")
        all_rankings[key_name] = get_rankings_for_pillar(
            df_current_year,
            df_previous_year_ranked,
            df_10_year_ago_ranked,
            pillar_col_name
        )
        logger.debug(f"Finished calculating {key_name}.")

    logger.info(f"Successfully generated all yearly rankings for {year}.")
    return all_rankings, message

if __name__ == '__main__':
    # Example usage for testing
    logger.info("Testing ranking_generator.py standalone")
    
    # Create dummy CSV for testing if it doesn't exist
    dummy_data_path = os.path.join(DATA_DIR, ANNUAL_SCORES_FILE)
    if not os.path.exists(dummy_data_path):
        logger.info(f"Creating dummy {ANNUAL_SCORES_FILE} for testing.")
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        dummy_data = {
            'Year': [2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022],
            'Country Name': ['Alpha', 'Bravo', 'Charlie', 'Alpha', 'Bravo', 'Charlie', 'Alpha', 'Bravo', 'Charlie'],
            'Pillar 1 Score': [80, 70, 90, 85, 72, 88, 82, 75, np.nan],
            'Pillar 2 Score': [75, 85, 80, 70, 88, 82, 77, 80, 90],
            'Pillar 3 Score': [90, 60, 70, 92, 65, 75, 90, 68, 72],
            'Total Index Average': [81.67, 71.67, 80.00, 82.33, 75.00, 81.67, 83.00, 74.33, 81.00] # Example averages
        }
        dummy_df = pd.DataFrame(dummy_data)
        # Ensure consistent column naming as expected by standardize_col_names
        dummy_df.columns = [col.replace('_', ' ').title() for col in dummy_df.columns]
        dummy_df.to_csv(dummy_data_path, index=False)
        logger.info(f"Dummy data written to {dummy_data_path}")


    try:
        test_year = 2021
        rankings, msg = generate_yearly_rankings(test_year)
        logger.info(f"Rankings for {test_year}:")
        # Pretty print the JSON-like structure
        import json
        logger.info(json.dumps(rankings, indent=2, default=str)) # Use default=str for NaT or other non-serializable
        if msg:
            logger.info(f"Message: {msg}")

        test_year_no_prev = 2020 # Year with no preceding year in dummy data
        rankings_no_prev, msg_no_prev = generate_yearly_rankings(test_year_no_prev)
        logger.info(f"Rankings for {test_year_no_prev} (no previous year data):")
        logger.info(json.dumps(rankings_no_prev, indent=2, default=str))
        if msg_no_prev:
            logger.info(f"Message: {msg_no_prev}")

        test_year_with_nan = 2022 # Year with NaN value in dummy data
        rankings_with_nan, msg_with_nan = generate_yearly_rankings(test_year_with_nan)
        logger.info(f"Rankings for {test_year_with_nan} (with NaN):")
        logger.info(json.dumps(rankings_with_nan, indent=2, default=str))
        if msg_with_nan:
            logger.info(f"Message: {msg_with_nan}")


    except Exception as e:
        logger.error(f"Error during standalone test: {e}", exc_info=True)
    
    # Example of how to trigger FileNotFoundError (assuming the file is there for the above)
    # try:
    #     os.rename(dummy_data_path, dummy_data_path + ".bak")
    #     generate_yearly_rankings(2021)
    # except FileNotFoundError as e:
    #     logger.error(f"Successfully caught expected FileNotFoundError: {e}")
    # finally:
    #     if os.path.exists(dummy_data_path + ".bak"):
    #         os.rename(dummy_data_path + ".bak", dummy_data_path) 