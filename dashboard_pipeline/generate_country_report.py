#!/usr/bin/env python3
"""Generates a JSON report for a specific country and time period."""

import pandas as pd
import numpy as np
import os
import argparse
import logging
import json
import re
from datetime import datetime

# --- Local Imports ---
from country_iso_map import COUNTRY_TO_ISO3 

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CURRENT_YEAR = datetime.now().year
MIN_YEAR = 1946
DECIMAL_PLACES = 2 # Define rounding precision

# --- Helper Functions ---
def safe_get_value(df, year, country_col, target_country, value_col):
    """Safely get a single value from a DataFrame for a specific year and country."""
    try:
        # Use .loc for label-based lookup, more explicit
        value = df.loc[(df['year'] == year) & (df[country_col] == target_country), value_col].iloc[0]
        # Check for pandas NA before converting to float or int
        # Return as Int if column name contains 'rank' and value is not NA, else float/None
        if pd.notna(value):
            if 'rank' in value_col:
                try:
                    return int(value)
                except (ValueError, TypeError):
                     logging.warning(f"Could not convert rank {value} to int for {value_col} in {year} for {target_country}.")
                     return None # Or keep as float?
            else:
                 return float(value)
        else:
            return None
    except IndexError:
        return None # No data for this specific year/country combination
    except KeyError:
         logging.warning(f"Column {value_col} not found while trying to get value for {year} for {target_country}.")
         return None
    except Exception as e:
        logging.warning(f"Error getting value for {value_col} in {year} for {target_country}: {e}")
        return None

def calculate_perc_change(start_val, end_val):
    """Calculate percentage change, handling None and zero start_val."""
    if start_val is None or end_val is None:
        return None
    if start_val == 0:
        # Decide handling: None is safer than potentially infinite or large numbers
        return None 
    try:
        change = ((end_val - start_val) / abs(start_val)) * 100
        return round(change, DECIMAL_PLACES)
    except Exception as e:
        logging.warning(f"Error calculating percentage change from {start_val} to {end_val}: {e}")
        return None

def standardize_col_names(df):
    """Converts DataFrame column names to lowercase snake_case."""
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    return df

def dataframe_to_json_list(df):
    """Converts relevant DataFrame columns to a list of dicts, handling NaN->None."""
    # Replace pandas NaN/NaT with Python None for JSON compatibility
    df_cleaned = df.replace({np.nan: None, pd.NaT: None})
    # Round numeric columns before converting to dict, handle ranks separately
    for col in df_cleaned.select_dtypes(include=np.number).columns:
        if 'rank' in col:
            # Attempt to convert ranks to nullable integers if possible, else leave as is (likely float with None)
            df_cleaned[col] = df_cleaned[col].astype(pd.Int64Dtype())
        else:
            df_cleaned[col] = df_cleaned[col].round(DECIMAL_PLACES)
    return df_cleaned.to_dict(orient='records')

def scale_similarity(score):
    """Scales cosine similarity (-1 to 1) to 0-100."""
    if pd.isna(score): 
        return None
    try:    
        # Clip ensures result is strictly within 0-100 even with potential float inaccuracies
        scaled = np.clip(((float(score) + 1) / 2) * 100, 0, 100)
        return round(scaled, DECIMAL_PLACES)
    except (ValueError, TypeError):
        return None

# --- Main Logic ---
def generate_report(country_iso, start_year, end_year, script_dir):
    """
    Generates the JSON report by loading data, calculating metrics, and writing output.
    """
    logging.info(f"Generating report for {country_iso} from {start_year} to {end_year}.")

    # --- Define File Paths (Using ./output consistently) ---
    output_subdir = os.path.join(script_dir, "JSON_output")
    annual_scores_path = os.path.join(script_dir, "dashboard_output", "annual_scores.csv")
    similarity_path = os.path.join(script_dir, "dashboard_output", "pairwise_similarity_yearly.csv")
    topic_votes_path = os.path.join(script_dir, "dashboard_output", "topic_votes_yearly.csv")
    report_filename = f"REPORT_{country_iso}_{start_year}-{end_year}.json"
    report_output_path = os.path.join(output_subdir, report_filename)

    # --- Load Data ---
    try:
        logging.info(f"Loading data from {annual_scores_path}")
        # Read, then standardize column names immediately
        df_scores_raw = pd.read_csv(annual_scores_path)
        df_scores = standardize_col_names(df_scores_raw.copy()) # Work on a copy
        
        logging.info(f"Loading data from {similarity_path}")
        df_similarity_raw = pd.read_csv(similarity_path)
        df_similarity = standardize_col_names(df_similarity_raw.copy())

        logging.info(f"Loading data from {topic_votes_path}")
        df_topics_raw = pd.read_csv(topic_votes_path)
        df_topics = standardize_col_names(df_topics_raw.copy())

    except FileNotFoundError as e:
        logging.error(f"ERROR: Input file not found: {e}. Ensure prerequisite scripts have run.")
        return
    except Exception as e:
        logging.error(f"ERROR: Failed to load input CSV files: {e}")
        return

    # Determine actual country column name after standardization
    country_col_scores = 'country_name' if 'country_name' in df_scores.columns else 'country'
    if country_col_scores not in df_scores.columns:
        logging.error(f"ERROR: Cannot find country identifier column in {annual_scores_path}")
        return
    country_col_topics = 'country' # As standardized from aggregate_topic_votes.py
    country_col_sim = 'country1_iso3' # One of the standardized similarity cols
    
    # Check if target country exists in scores data at all
    if country_iso not in df_scores[country_col_scores].unique():
        logging.error(f"ERROR: Country ISO {country_iso} not found in the scores data file {annual_scores_path}.")
        return
        
    # --- Filter Data For Country & Period ---
    logging.info("Filtering data for the specified country and period...")
    # Ensure 'year' column is numeric before filtering
    df_scores['year'] = pd.to_numeric(df_scores['year'], errors='coerce')
    df_scores.dropna(subset=['year'], inplace=True) 
    df_scores['year'] = df_scores['year'].astype(int) 
    
    df_similarity['year'] = pd.to_numeric(df_similarity['year'], errors='coerce')
    df_similarity.dropna(subset=['year'], inplace=True)
    df_similarity['year'] = df_similarity['year'].astype(int)
    
    df_topics['year'] = pd.to_numeric(df_topics['year'], errors='coerce')
    df_topics.dropna(subset=['year'], inplace=True)
    df_topics['year'] = df_topics['year'].astype(int)
    
    # Filter by time range first for global calculations
    df_scores_period = df_scores[(df_scores['year'] >= start_year) & (df_scores['year'] <= end_year)].copy()
    df_topics_period = df_topics[(df_topics['year'] >= start_year) & (df_topics['year'] <= end_year)].copy()
    
    # Filter for the specific country
    df_scores_country = df_scores_period[df_scores_period[country_col_scores] == country_iso].copy()
    if df_scores_country.empty:
        logging.error(f"ERROR: No score data found for country {country_iso} within the period {start_year}-{end_year}.")
        return
        
    df_similarity_country = df_similarity[
        ((df_similarity[country_col_sim] == country_iso) | (df_similarity['country2_iso3'] == country_iso)) & 
        (df_similarity['year'] >= start_year) & 
        (df_similarity['year'] <= end_year)
    ].copy()

    df_topics_country = df_topics_period[df_topics_period[country_col_topics] == country_iso].copy()

    # --- Calculations ---
    results = {}

    # --- Create Reverse Mapping (ISO3 -> Country Name) ---
    ISO3_TO_COUNTRY = {v: k for k, v in COUNTRY_TO_ISO3.items()}

    # --- Add Report Metadata --- 
    logging.info("Adding report metadata...")
    country_name = ISO3_TO_COUNTRY.get(country_iso, country_iso) # Fallback to ISO if not found
    results['report_metadata'] = {
        'country_iso3': country_iso, 
        'country_name': country_name, # Added full country name
        'start_year': start_year,
        'end_year': end_year
    }
    
    # --- Calculate World Averages for the Period --- (Keep this section for overall summary)
    logging.info("Calculating world average scores for the period...")
    world_avg_scores = {}
    world_avg_cols = ['pillar_1_score', 'pillar_2_score', 'pillar_3_score', 'total_index_average']
    for col in world_avg_cols:
        if col in df_scores_period.columns:
            # Calculate mean, dropna=True is default but explicit
            world_avg = df_scores_period[col].mean(skipna=True) 
            world_avg_scores[f'world_avg_{col}'] = round(world_avg, DECIMAL_PLACES) if pd.notna(world_avg) else None
        else:
            world_avg_scores[f'world_avg_{col}'] = None
            logging.warning(f"World average column '{col}' not found in scores data.")
    results['world_average_scores_period'] = world_avg_scores
    # --- End of Section ---

    # 1. Index Score + Change + Rank
    logging.info("Calculating Index Score change and ranks...")
    start_index_score = safe_get_value(df_scores_country, start_year, country_col_scores, country_iso, 'total_index_average')
    end_index_score = safe_get_value(df_scores_country, end_year, country_col_scores, country_iso, 'total_index_average')
    start_rank = safe_get_value(df_scores_country, start_year, country_col_scores, country_iso, 'overall_rank') # Get start rank
    end_rank = safe_get_value(df_scores_country, end_year, country_col_scores, country_iso, 'overall_rank')       # Get end rank
    index_perc_change = calculate_perc_change(start_index_score, end_index_score)
    results['index_score_analysis'] = {
        'end_year_score': round(end_index_score, DECIMAL_PLACES) if end_index_score is not None else None,
        'start_year_score': round(start_index_score, DECIMAL_PLACES) if start_index_score is not None else None,
        'percentage_change': index_perc_change, # Already rounded
        'start_year_rank': start_rank, # Added start rank (already int or None)
        'end_year_rank': end_rank     # Added end rank (already int or None)
    }

    # 2. Vote Totals & % (Country vs World)
    logging.info("Calculating overall vote behavior (Country vs World)...")
    # Country Totals
    country_total_yes = df_scores_country['yes_votes'].sum()
    country_total_no = df_scores_country['no_votes'].sum()
    country_total_abstain = df_scores_country['abstain_votes'].sum()
    country_total_votes = country_total_yes + country_total_no + country_total_abstain
    # Country Percentages
    country_yes_perc = (country_total_yes / country_total_votes * 100) if country_total_votes > 0 else 0
    country_no_perc = (country_total_no / country_total_votes * 100) if country_total_votes > 0 else 0
    country_abs_perc = (country_total_abstain / country_total_votes * 100) if country_total_votes > 0 else 0
    
    # World Totals (for the period)
    # Ensure columns exist before summing
    world_total_yes = df_scores_period['yes_votes'].sum() if 'yes_votes' in df_scores_period else 0
    world_total_no = df_scores_period['no_votes'].sum() if 'no_votes' in df_scores_period else 0
    world_total_abstain = df_scores_period['abstain_votes'].sum() if 'abstain_votes' in df_scores_period else 0
    world_total_votes = world_total_yes + world_total_no + world_total_abstain
    # World Percentages
    world_yes_perc = (world_total_yes / world_total_votes * 100) if world_total_votes > 0 else 0
    world_no_perc = (world_total_no / world_total_votes * 100) if world_total_votes > 0 else 0
    world_abs_perc = (world_total_abstain / world_total_votes * 100) if world_total_votes > 0 else 0
    
    results['voting_behavior_overall'] = {
        'total_votes': int(country_total_votes),
        'yes_votes': int(country_total_yes),
        'no_votes': int(country_total_no),
        'abstain_votes': int(country_total_abstain),
        'yes_percentage': round(country_yes_perc, DECIMAL_PLACES),
        'no_percentage': round(country_no_perc, DECIMAL_PLACES),
        'abstain_percentage': round(country_abs_perc, DECIMAL_PLACES),
        # Percentage point difference vs world average
        'yes_vs_world_avg': round(country_yes_perc - world_yes_perc, DECIMAL_PLACES),
        'no_vs_world_avg': round(country_no_perc - world_no_perc, DECIMAL_PLACES),
        'abstain_vs_world_avg': round(country_abs_perc - world_abs_perc, DECIMAL_PLACES)
    }

    # 3. Pillar Scores Time Series (ADDING YEARLY WORLD AVERAGES)
    logging.info("Extracting scores timeseries and calculating yearly world averages...")
    
    # --- Calculate Yearly World Averages --- (NEW STEP)
    world_avg_cols = ['pillar_1_score', 'pillar_2_score', 'pillar_3_score', 'total_index_average']
    world_avg_cols_present = [col for col in world_avg_cols if col in df_scores_period.columns]
    df_world_yearly_avg = pd.DataFrame() # Initialize empty df
    if world_avg_cols_present:
        try:
            # Calculates the mean for EACH column in world_avg_cols_present, grouped by year
            # REMOVED skipna=True as it's default for groupby().mean() and caused TypeError
            df_world_yearly_avg = df_scores_period.groupby('year')[world_avg_cols_present].mean().reset_index()
            # Renames columns like 'pillar_1_score' to 'world_avg_pillar_1_score' etc.
            df_world_yearly_avg.columns = ['year'] + [f'world_avg_{col}' for col in world_avg_cols_present]
            logging.info("Calculated yearly world average scores.")
        except Exception as e:
            logging.warning(f"Could not calculate yearly world averages: {e}")
            df_world_yearly_avg = pd.DataFrame(columns=['year'])
    else:
        logging.warning("No score columns found to calculate yearly world averages.")
        df_world_yearly_avg = pd.DataFrame(columns=['year'])
    # --- End of New Step ---
        
    # Select country's scores and ranks
    score_cols = [
        'year', 
        'pillar_1_score', 'pillar_2_score', 'pillar_3_score', 'total_index_average',
        'pillar_1_rank', 'pillar_2_rank', 'pillar_3_rank', 'overall_rank'
    ]
    score_cols_present = [col for col in score_cols if col in df_scores_country.columns]
    df_scores_ts = df_scores_country[score_cols_present].sort_values(by='year')

    # Merge country data with yearly world averages
    if not df_world_yearly_avg.empty:
        # This merge adds the 'world_avg_pillar_1_score', 'world_avg_pillar_2_score', etc. columns
        df_scores_ts_merged = pd.merge(df_scores_ts, df_world_yearly_avg, on='year', how='left')
    else:
        df_scores_ts_merged = df_scores_ts
        logging.warning("Proceeding without merging yearly world averages.")

    # --- Remove Debug Logging ---
    # logging.debug(f"Columns in df_scores_ts_merged before converting to JSON: {df_scores_ts_merged.columns.tolist()}")
    # --- End Debug Logging ---

    # Rounding and NaN->None applied by dataframe_to_json_list
    results['scores_timeseries'] = dataframe_to_json_list(df_scores_ts_merged) # Use the merged dataframe

    # 4. & 5. Allies & Enemies (Scaled Score)
    logging.info("Calculating Allies and Enemies...")
    allies = []
    enemies = []
    if not df_similarity_country.empty:
        # Determine the 'other' country in each pair
        df_similarity_country['other_country'] = df_similarity_country.apply(
            lambda row: row['country2_iso3'] if row[country_col_sim] == country_iso else row[country_col_sim],
            axis=1
        )
        # Calculate average similarity over the period
        # Use standardized column name 'cosinesimilarity' -> 'cosine_similarity'
        avg_similarity = df_similarity_country.groupby('other_country')['cosinesimilarity'].mean().reset_index()
        avg_similarity = avg_similarity.rename(columns={'other_country': 'country', 'cosinesimilarity': 'average_similarity_score'})
        
        # Scale the similarity score
        avg_similarity['average_similarity_score_scaled'] = avg_similarity['average_similarity_score'].apply(scale_similarity)
        
        # Sort for allies and enemies using SCALED score
        allies_df = avg_similarity.sort_values(by='average_similarity_score_scaled', ascending=False).head(5)
        enemies_df = avg_similarity.sort_values(by='average_similarity_score_scaled', ascending=True).head(5)
        
        # Select columns for output, rounding applied by dataframe_to_json_list
        allies = dataframe_to_json_list(allies_df[['country', 'average_similarity_score_scaled']])
        enemies = dataframe_to_json_list(enemies_df[['country', 'average_similarity_score_scaled']])
    else:
        logging.warning(f"No similarity data found for {country_iso} in the period.")
    results['top_allies'] = allies # Standardized key
    results['top_enemies'] = enemies # Standardized key

    # 6. & 7. Top Topics & All Topics (vs World)
    logging.info("Calculating Topic Voting (Country vs World)...")
    top_supported = []
    top_opposed = []
    all_topic_voting = []
    
    # --- World Topic Aggregation (for the period) ---
    df_topic_agg_world = df_topics_period.groupby('topictag').agg(
            world_total_yes=('yesvotes_topic', 'sum'),
            world_total_no=('novotes_topic', 'sum'),
            world_total_abstain=('abstainvotes_topic', 'sum'), # Added Abstain
            world_total_votes=('totalvotes_topic', 'sum')
        ).reset_index()
    df_topic_agg_world = df_topic_agg_world[df_topic_agg_world['world_total_votes'] > 0].copy()
    if not df_topic_agg_world.empty:
        df_topic_agg_world['world_support_percentage'] = (df_topic_agg_world['world_total_yes'] / df_topic_agg_world['world_total_votes']) * 100
        df_topic_agg_world['world_opposition_percentage'] = (df_topic_agg_world['world_total_no'] / df_topic_agg_world['world_total_votes']) * 100
        df_topic_agg_world['world_abstain_percentage'] = (df_topic_agg_world['world_total_abstain'] / df_topic_agg_world['world_total_votes']) * 100 # Added Abstain %
    else: 
        logging.warning("No world topic votes found for the period.")
        # Create empty df with expected columns to prevent merge errors
        df_topic_agg_world = pd.DataFrame(columns=[
            'topictag', 'world_total_yes', 'world_total_no', 'world_total_abstain',
            'world_total_votes', 'world_support_percentage', 'world_opposition_percentage', 'world_abstain_percentage'
            ])
        
    # --- Country Topic Aggregation & Comparison ---
    if not df_topics_country.empty:
        # Aggregate country topic votes
        df_topic_agg_country = df_topics_country.groupby('topictag').agg(
            total_yes=('yesvotes_topic', 'sum'),
            total_no=('novotes_topic', 'sum'),
            total_abstain=('abstainvotes_topic', 'sum'), # Added Abstain
            total_votes=('totalvotes_topic', 'sum')
        ).reset_index()

        # Filter out topics country didn't vote on
        df_topic_agg_country = df_topic_agg_country[df_topic_agg_country['total_votes'] > 0].copy()

        if not df_topic_agg_country.empty:
            # Calculate country percentages
            df_topic_agg_country['support_percentage'] = (df_topic_agg_country['total_yes'] / df_topic_agg_country['total_votes']) * 100
            df_topic_agg_country['opposition_percentage'] = (df_topic_agg_country['total_no'] / df_topic_agg_country['total_votes']) * 100
            df_topic_agg_country['abstain_percentage'] = (df_topic_agg_country['total_abstain'] / df_topic_agg_country['total_votes']) * 100 # Added Abstain %
            
            # Merge world averages
            df_topic_combined = pd.merge(
                df_topic_agg_country,
                df_topic_agg_world[['topictag', 'world_support_percentage', 'world_opposition_percentage', 'world_abstain_percentage']], # Added world Abstain %
                on='topictag',
                how='left' # Keep all country topics, add world avg where available
            )
            
            # Calculate differences vs world average (percentage point difference)
            df_topic_combined['support_vs_world_avg'] = df_topic_combined['support_percentage'] - df_topic_combined['world_support_percentage']
            df_topic_combined['opposition_vs_world_avg'] = df_topic_combined['opposition_percentage'] - df_topic_combined['world_opposition_percentage']
            df_topic_combined['abstain_vs_world_avg'] = df_topic_combined['abstain_percentage'] - df_topic_combined['world_abstain_percentage'] # Added Abstain Diff
            
            # --- Prepare Output --- 
            # ALL Topics Output (Ensure all necessary columns are included)
            all_topic_output_cols = [
                'topictag', 
                'support_percentage', 'opposition_percentage', 'abstain_percentage', 
                'world_support_percentage', 'world_opposition_percentage', 'world_abstain_percentage', # Added World % for context
                'support_vs_world_avg', 'opposition_vs_world_avg', 'abstain_vs_world_avg', 
                'total_yes', 'total_no', 'total_abstain', 'total_votes' # Added total_abstain
            ]
            all_topic_output_cols_present = [col for col in all_topic_output_cols if col in df_topic_combined.columns]
            all_topic_voting = dataframe_to_json_list(df_topic_combined[all_topic_output_cols_present])

            # TOP 3 Supported (based on country percentage)
            df_supported = df_topic_combined.sort_values(by='support_percentage', ascending=False).head(3)
            # REVISED output columns for clarity
            top_supported_output_cols = [
                'topictag', 
                'support_percentage', 
                'world_support_percentage', # Added World Average
                'support_vs_world_avg', 
                'total_yes', # Added Yes Count
                'total_votes' # Added Total Votes Count
            ]
            top_supported_output_cols_present = [col for col in top_supported_output_cols if col in df_supported.columns]
            top_supported = dataframe_to_json_list(df_supported[top_supported_output_cols_present])
            
            # TOP 3 Opposed (based on country percentage)
            df_opposed = df_topic_combined.sort_values(by='opposition_percentage', ascending=False).head(3)
            # REVISED output columns for clarity
            top_opposed_output_cols = [
                'topictag', 
                'opposition_percentage', 
                'world_opposition_percentage', # Added World Average
                'opposition_vs_world_avg', 
                'total_no', # Added No Count
                'total_votes' # Added Total Votes Count
            ]
            top_opposed_output_cols_present = [col for col in top_opposed_output_cols if col in df_opposed.columns]
            top_opposed = dataframe_to_json_list(df_opposed[top_opposed_output_cols_present])
        else:
             logging.warning(f"No topics with votes > 0 found for {country_iso} in the period after aggregation.")
    else:
        logging.warning(f"No topic vote data found for {country_iso} in the period.")
        
    results['top_supported_topics'] = top_supported # Standardized key
    results['top_opposed_topics'] = top_opposed   # Standardized key
    results['all_topic_voting'] = all_topic_voting # New key

    # --- Write JSON Output ---
    logging.info(f"Writing report to {report_output_path}")
    # Ensure output directory exists
    os.makedirs(output_subdir, exist_ok=True)
    try:
        with open(report_output_path, 'w') as f:
            # Use default=str to handle potential non-serializable types defensively (e.g. pandas Int64)
            json.dump(results, f, indent=4, default=str) 
        logging.info("Report generated successfully.")
    except Exception as e:
        logging.error(f"ERROR: Failed to write JSON report: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Generate a JSON report for UN voting patterns of a specific country over a time period.")
    parser.add_argument("country_iso", type=str, help="3-letter ISO code of the country (e.g., USA).")
    parser.add_argument("start_year", type=int, help=f"Start year of the period (inclusive, >= {MIN_YEAR}).")
    parser.add_argument("end_year", type=int, help=f"End year of the period (inclusive, <= {CURRENT_YEAR}).")

    args = parser.parse_args()

    # Input Validation
    valid_input = True
    if not re.match(r"^[A-Z]{3}$", args.country_iso):
        logging.error("Invalid country ISO code format. Please use 3 uppercase letters (e.g., USA).")
        valid_input = False
    if not (MIN_YEAR <= args.start_year <= CURRENT_YEAR):
        logging.error(f"Start year must be between {MIN_YEAR} and {CURRENT_YEAR}.")
        valid_input = False
    if not (MIN_YEAR <= args.end_year <= CURRENT_YEAR):
        logging.error(f"End year must be between {MIN_YEAR} and {CURRENT_YEAR}.")
        valid_input = False
    if args.end_year < args.start_year:
         logging.error("End year cannot be before start year.")
         valid_input = False
    # Requirement: at least two years separating start and end (inclusive range size >= 3)
    if (args.end_year - args.start_year) < 2:
        logging.error("The period must span at least 3 years (end_year - start_year >= 2).")
        valid_input = False

    if valid_input:
        generate_report(args.country_iso, args.start_year, args.end_year, script_dir)
    else:
        logging.error("Script aborted due to invalid input.")

    logging.info("Script finished.") 