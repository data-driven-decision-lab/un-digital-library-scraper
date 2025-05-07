#!/usr/bin/env python3
"""Generates a JSON report for a specific country and time period.

This script is designed to be called as a module by the FastAPI application.
It expects data CSV files to be in a 'dashboard_output' subdirectory
relative to this script's location.
"""

import pandas as pd
import numpy as np
import os
import logging # Keep logging for messages, but API will configure handlers
import json # Though not used for writing files anymore, kept for potential internal use
import re
from datetime import datetime

# --- Local Imports (relative for package structure) ---
from .country_iso_map import COUNTRY_TO_ISO3

# --- Configuration ---
# Logging will be configured by the calling application (FastAPI)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "required_csvs")

# Year constraints for data consistency if used internally by this script
# Primary validation for API parameters will be in the FastAPI layer.
MIN_DATA_YEAR = 1946
MAX_DATA_YEAR = 2024 # Updated to fixed max year as per user request
DECIMAL_PLACES = 2

# --- Helper Functions (copied from original, unchanged) ---
def safe_get_value(df, year, country_col, target_country, value_col):
    """Safely get a single value from a DataFrame for a specific year and country."""
    try:
        value = df.loc[(df['year'] == year) & (df[country_col] == target_country), value_col].iloc[0]
        if pd.notna(value):
            if 'rank' in value_col:
                try:
                    return int(value)
                except (ValueError, TypeError):
                     logging.warning(f"Could not convert rank {value} to int for {value_col} in {year} for {target_country}.")
                     return None
            else:
                 return float(value)
        else:
            return None
    except IndexError:
        return None
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
    df_cleaned = df.replace({np.nan: None, pd.NaT: None})
    for col in df_cleaned.select_dtypes(include=np.number).columns:
        if 'rank' in col:
            df_cleaned[col] = df_cleaned[col].astype(pd.Int64Dtype())
        else:
            df_cleaned[col] = df_cleaned[col].round(DECIMAL_PLACES)
    return df_cleaned.to_dict(orient='records')

def scale_similarity(score):
    """Scales cosine similarity (-1 to 1) to 0-100."""
    if pd.isna(score): 
        return None
    try:    
        scaled = np.clip(((float(score) + 1) / 2) * 100, 0, 100)
        return round(scaled, DECIMAL_PLACES)
    except (ValueError, TypeError):
        return None

# --- Main Report Generation Logic ---
def generate_report(country_iso: str, start_year: int, end_year: int) -> dict:
    """
    Generates the JSON-like report data as a dictionary.
    Raises FileNotFoundError if data files are missing.
    Raises ValueError if data for the specified country/period is not found or invalid.
    """
    logging.info(f"Report generation started for {country_iso}, period: {start_year}-{end_year}.")

    # --- Define File Paths (relative to this script's 'dashboard_output' subdir) ---
    annual_scores_path = os.path.join(DATA_DIR, "annual_scores.csv")
    similarity_path = os.path.join(DATA_DIR, "pairwise_similarity_yearly.csv")
    topic_votes_path = os.path.join(DATA_DIR, "topic_votes_yearly.csv")

    # --- Load Data ---
    try:
        logging.debug(f"Loading data from {annual_scores_path}")
        df_scores_raw = pd.read_csv(annual_scores_path)
        df_scores = standardize_col_names(df_scores_raw.copy())
        
        logging.debug(f"Loading data from {similarity_path}")
        df_similarity_raw = pd.read_csv(similarity_path)
        df_similarity = standardize_col_names(df_similarity_raw.copy())

        logging.debug(f"Loading data from {topic_votes_path}")
        df_topics_raw = pd.read_csv(topic_votes_path)
        df_topics = standardize_col_names(df_topics_raw.copy())

    except FileNotFoundError as e:
        logging.error(f"Input data file not found: {e.filename}. Ensure CSV files are in {DATA_DIR}")
        raise # Re-raise for the API to catch and return a proper HTTP error
    except Exception as e:
        logging.error(f"Failed to load or standardize input CSV files: {e}")
        raise ValueError(f"Error processing input data files: {e}")

    # Determine country column names (assuming standardized names from CSVs)
    country_col_scores = 'country_name' # Expects 'country_name' to be the ISO3 code in annual_scores.csv
    if country_col_scores not in df_scores.columns:
        # Fallback or error if 'country_name' (as ISO3) isn't there.
        # This depends on the actual content of your annual_scores.csv
        # If 'country' is the ISO3 column and 'country_name' is the full name, adjust accordingly.
        # For now, assuming 'country_name' IS the ISO3 code column in this file after standardization.
        alt_country_col = 'country'
        if alt_country_col in df_scores.columns:
            country_col_scores = alt_country_col
            logging.warning(f"Using '{alt_country_col}' as country ISO column in scores data.")
        else:
             raise ValueError(f"Cannot find a suitable country ISO identifier column (expected '{country_col_scores}' or '{alt_country_col}') in {annual_scores_path}")

    country_col_topics = 'country' # Expected ISO3 column in topic_votes_yearly.csv
    country_col_sim = 'country1_iso3' # Expected one of the ISO3 columns in pairwise_similarity_yearly.csv

    if country_iso not in df_scores[country_col_scores].unique():
        raise ValueError(f"Country ISO '{country_iso}' not found in the scores data ({annual_scores_path}).")
        
    # --- Filter Data For Country & Period ---
    df_scores['year'] = pd.to_numeric(df_scores['year'], errors='coerce')
    df_scores.dropna(subset=['year'], inplace=True)
    df_scores['year'] = df_scores['year'].astype(int)
    
    df_similarity['year'] = pd.to_numeric(df_similarity['year'], errors='coerce')
    df_similarity.dropna(subset=['year'], inplace=True)
    df_similarity['year'] = df_similarity['year'].astype(int)
    
    df_topics['year'] = pd.to_numeric(df_topics['year'], errors='coerce')
    df_topics.dropna(subset=['year'], inplace=True)
    df_topics['year'] = df_topics['year'].astype(int)
    
    df_scores_period = df_scores[(df_scores['year'] >= start_year) & (df_scores['year'] <= end_year)].copy()
    df_topics_period = df_topics[(df_topics['year'] >= start_year) & (df_topics['year'] <= end_year)].copy()
    
    df_scores_country = df_scores_period[df_scores_period[country_col_scores] == country_iso].copy()
    if df_scores_country.empty:
        raise ValueError(f"No score data found for country '{country_iso}' within the period {start_year}-{end_year}. Check data and country ISO column ('{country_col_scores}').")
        
    df_similarity_country = df_similarity[
        ((df_similarity[country_col_sim] == country_iso) | (df_similarity['country2_iso3'] == country_iso)) &
        (df_similarity['year'] >= start_year) & (df_similarity['year'] <= end_year)
    ].copy()

    df_topics_country = df_topics_period[df_topics_period[country_col_topics] == country_iso].copy()

    results = {}

    # --- Create Reverse Mapping (ISO3 -> Country Name from the imported dict) ---
    # The imported COUNTRY_TO_ISO3 is Name: ISO. We need ISO: Name.
    ISO3_TO_COUNTRY = {v: k for k, v in COUNTRY_TO_ISO3.items()}

    # --- Add Report Metadata ---
    logging.debug("Adding report metadata...")
    # Fallback to ISO code itself if not found in the map, though this should be rare.
    full_country_name = ISO3_TO_COUNTRY.get(country_iso, country_iso) 
    results['report_metadata'] = {
        'country_iso3': country_iso,
        'country_name': full_country_name,
        'start_year': start_year,
        'end_year': end_year
    }
    
    # --- Calculate World Averages for the Period --- 
    logging.debug("Calculating world average scores for the period...")
    world_avg_scores_period_data = {}
    world_avg_cols_period = ['pillar_1_score', 'pillar_2_score', 'pillar_3_score', 'total_index_average']
    for col in world_avg_cols_period:
        if col in df_scores_period.columns:
            world_avg = df_scores_period[col].mean() # skipna=True is default for Series.mean()
            world_avg_scores_period_data[f'world_avg_{col}'] = round(world_avg, DECIMAL_PLACES) if pd.notna(world_avg) else None
        else:
            world_avg_scores_period_data[f'world_avg_{col}'] = None
            logging.warning(f"World average column '{col}' for period calculation not found in scores data.")
    results['world_average_scores_period'] = world_avg_scores_period_data

    # --- Index Score Analysis ---
    logging.debug("Calculating Index Score change and ranks...")
    start_index_score = safe_get_value(df_scores_country, start_year, country_col_scores, country_iso, 'total_index_average')
    end_index_score = safe_get_value(df_scores_country, end_year, country_col_scores, country_iso, 'total_index_average')
    start_rank = safe_get_value(df_scores_country, start_year, country_col_scores, country_iso, 'overall_rank')
    end_rank = safe_get_value(df_scores_country, end_year, country_col_scores, country_iso, 'overall_rank')
    index_perc_change = calculate_perc_change(start_index_score, end_index_score)
    results['index_score_analysis'] = {
        'end_year_score': round(end_index_score, DECIMAL_PLACES) if end_index_score is not None else None,
        'start_year_score': round(start_index_score, DECIMAL_PLACES) if start_index_score is not None else None,
        'percentage_change': index_perc_change,
        'start_year_rank': start_rank,
        'end_year_rank': end_rank
    }

    # --- Overall Voting Behavior (Country vs World) ---
    logging.debug("Calculating overall vote behavior (Country vs World)...")
    country_total_yes = df_scores_country['yes_votes'].sum()
    country_total_no = df_scores_country['no_votes'].sum()
    country_total_abstain = df_scores_country['abstain_votes'].sum()
    country_total_votes = country_total_yes + country_total_no + country_total_abstain
    
    country_yes_perc = (country_total_yes / country_total_votes * 100) if country_total_votes > 0 else 0
    country_no_perc = (country_total_no / country_total_votes * 100) if country_total_votes > 0 else 0
    country_abs_perc = (country_total_abstain / country_total_votes * 100) if country_total_votes > 0 else 0
    
    world_total_yes = df_scores_period['yes_votes'].sum() if 'yes_votes' in df_scores_period else 0
    world_total_no = df_scores_period['no_votes'].sum() if 'no_votes' in df_scores_period else 0
    world_total_abstain = df_scores_period['abstain_votes'].sum() if 'abstain_votes' in df_scores_period else 0
    world_total_votes = world_total_yes + world_total_no + world_total_abstain
    
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
        'world_yes_percentage': round(world_yes_perc, DECIMAL_PLACES),
        'world_no_percentage': round(world_no_perc, DECIMAL_PLACES),
        'world_abstain_percentage': round(world_abs_perc, DECIMAL_PLACES),
        'yes_vs_world_avg': round(country_yes_perc - world_yes_perc, DECIMAL_PLACES),
        'no_vs_world_avg': round(country_no_perc - world_no_perc, DECIMAL_PLACES),
        'abstain_vs_world_avg': round(country_abs_perc - world_abs_perc, DECIMAL_PLACES)
    }

    # --- Pillar Scores Time Series (with Yearly World Averages) ---
    logging.debug("Extracting scores timeseries and calculating yearly world averages...")
    world_avg_cols_yearly = ['pillar_1_score', 'pillar_2_score', 'pillar_3_score', 'total_index_average']
    df_world_yearly_avg = pd.DataFrame()
    if all(col in df_scores_period.columns for col in world_avg_cols_yearly):
        try:
            df_world_yearly_avg = df_scores_period.groupby('year')[world_avg_cols_yearly].mean().reset_index()
            df_world_yearly_avg.columns = ['year'] + [f'world_avg_{col}' for col in world_avg_cols_yearly]
        except Exception as e:
            logging.warning(f"Could not calculate yearly world averages: {e}. Proceeding without them.")
            df_world_yearly_avg = pd.DataFrame(columns=['year'] + [f'world_avg_{col}' for col in world_avg_cols_yearly]) # empty with correct cols
    else:
        logging.warning("Not all columns for yearly world averages found. Proceeding without them.")
        df_world_yearly_avg = pd.DataFrame(columns=['year'] + [f'world_avg_{col}' for col in world_avg_cols_yearly])
        
    score_cols = [
        'year', 'pillar_1_score', 'pillar_2_score', 'pillar_3_score', 'total_index_average',
        'pillar_1_rank', 'pillar_2_rank', 'pillar_3_rank', 'overall_rank'
    ]
    score_cols_present = [col for col in score_cols if col in df_scores_country.columns]
    df_scores_ts = df_scores_country[score_cols_present].sort_values(by='year')

    if not df_world_yearly_avg.empty and 'year' in df_world_yearly_avg.columns:
        df_scores_ts_merged = pd.merge(df_scores_ts, df_world_yearly_avg, on='year', how='left')
    else: # Ensure all world_avg columns exist even if empty, for Pydantic model consistency
        df_scores_ts_merged = df_scores_ts.copy()
        for col in [f'world_avg_{c}' for c in world_avg_cols_yearly]:
            if col not in df_scores_ts_merged.columns:
                 df_scores_ts_merged[col] = None 
        logging.warning("Proceeding with scores_timeseries potentially missing some world average columns or all if calculation failed.")
        
    results['scores_timeseries'] = dataframe_to_json_list(df_scores_ts_merged)

    # --- Allies & Enemies ---
    logging.debug("Calculating Allies and Enemies...")
    allies = []
    enemies = []
    if not df_similarity_country.empty and 'cosinesimilarity' in df_similarity_country.columns:
        df_similarity_country['other_country'] = df_similarity_country.apply(
            lambda row: row['country2_iso3'] if row[country_col_sim] == country_iso else row[country_col_sim],
            axis=1
        )
        avg_similarity = df_similarity_country.groupby('other_country')['cosinesimilarity'].mean().reset_index()
        avg_similarity = avg_similarity.rename(columns={'other_country': 'country', 'cosinesimilarity': 'average_similarity_score'})
        avg_similarity['average_similarity_score_scaled'] = avg_similarity['average_similarity_score'].apply(scale_similarity)
        
        allies_df = avg_similarity.sort_values(by='average_similarity_score_scaled', ascending=False).head(5)
        enemies_df = avg_similarity.sort_values(by='average_similarity_score_scaled', ascending=True).head(5)
        
        allies = dataframe_to_json_list(allies_df[['country', 'average_similarity_score_scaled']])
        enemies = dataframe_to_json_list(enemies_df[['country', 'average_similarity_score_scaled']])
    else:
        logging.warning(f"No similarity data or 'cosinesimilarity' column found for {country_iso}. Allies/Enemies will be empty.")
    results['top_allies'] = allies
    results['top_enemies'] = enemies

    # --- Topic Voting (Country vs World) ---
    logging.debug("Calculating Topic Voting (Country vs World)...")
    top_supported = []
    top_opposed = []
    all_topic_voting = []
    
    df_topic_agg_world = df_topics_period.groupby('topictag').agg(
            world_total_yes=('yesvotes_topic', 'sum'),
            world_total_no=('novotes_topic', 'sum'),
            world_total_abstain=('abstainvotes_topic', 'sum'),
            world_total_votes=('totalvotes_topic', 'sum')
        ).reset_index()
    df_topic_agg_world = df_topic_agg_world[df_topic_agg_world['world_total_votes'] > 0].copy()

    if not df_topic_agg_world.empty:
        df_topic_agg_world['world_support_percentage'] = (df_topic_agg_world['world_total_yes'] / df_topic_agg_world['world_total_votes']) * 100
        df_topic_agg_world['world_opposition_percentage'] = (df_topic_agg_world['world_total_no'] / df_topic_agg_world['world_total_votes']) * 100
        df_topic_agg_world['world_abstain_percentage'] = (df_topic_agg_world['world_total_abstain'] / df_topic_agg_world['world_total_votes']) * 100
    else:
        logging.warning("No world topic votes for aggregation found for the period.")
        # Prepare empty df with expected columns for robust merge
        df_topic_agg_world = pd.DataFrame(columns=[
            'topictag', 'world_total_yes', 'world_total_no', 'world_total_abstain',
            'world_total_votes', 'world_support_percentage', 'world_opposition_percentage', 'world_abstain_percentage'
        ])
        
    if not df_topics_country.empty:
        df_topic_agg_country = df_topics_country.groupby('topictag').agg(
            total_yes=('yesvotes_topic', 'sum'),
            total_no=('novotes_topic', 'sum'),
            total_abstain=('abstainvotes_topic', 'sum'),
            total_votes=('totalvotes_topic', 'sum')
        ).reset_index()
        df_topic_agg_country = df_topic_agg_country[df_topic_agg_country['total_votes'] > 0].copy()

        if not df_topic_agg_country.empty:
            df_topic_agg_country['support_percentage'] = (df_topic_agg_country['total_yes'] / df_topic_agg_country['total_votes']) * 100
            df_topic_agg_country['opposition_percentage'] = (df_topic_agg_country['total_no'] / df_topic_agg_country['total_votes']) * 100
            df_topic_agg_country['abstain_percentage'] = (df_topic_agg_country['total_abstain'] / df_topic_agg_country['total_votes']) * 100
            
            df_topic_combined = pd.merge(
                df_topic_agg_country,
                df_topic_agg_world[['topictag', 'world_support_percentage', 'world_opposition_percentage', 'world_abstain_percentage']],
                on='topictag',
                how='left'
            )
            
            df_topic_combined['support_vs_world_avg'] = df_topic_combined['support_percentage'] - df_topic_combined['world_support_percentage']
            df_topic_combined['opposition_vs_world_avg'] = df_topic_combined['opposition_percentage'] - df_topic_combined['world_opposition_percentage']
            df_topic_combined['abstain_vs_world_avg'] = df_topic_combined['abstain_percentage'] - df_topic_combined['world_abstain_percentage']
            
            all_topic_output_cols = [
                'topictag', 'support_percentage', 'opposition_percentage', 'abstain_percentage',
                'world_support_percentage', 'world_opposition_percentage', 'world_abstain_percentage',
                'support_vs_world_avg', 'opposition_vs_world_avg', 'abstain_vs_world_avg',
                'total_yes', 'total_no', 'total_abstain', 'total_votes'
            ]
            all_topic_output_cols_present = [col for col in all_topic_output_cols if col in df_topic_combined.columns]
            all_topic_voting = dataframe_to_json_list(df_topic_combined[all_topic_output_cols_present])

            df_supported = df_topic_combined.sort_values(by='support_percentage', ascending=False).head(3)
            top_supported_output_cols = [
                'topictag', 'support_percentage', 'world_support_percentage',
                'support_vs_world_avg', 'total_yes', 'total_votes'
            ]
            top_supported_output_cols_present = [col for col in top_supported_output_cols if col in df_supported.columns]
            top_supported = dataframe_to_json_list(df_supported[top_supported_output_cols_present])
            
            df_opposed = df_topic_combined.sort_values(by='opposition_percentage', ascending=False).head(3)
            top_opposed_output_cols = [
                'topictag', 'opposition_percentage', 'world_opposition_percentage',
                'opposition_vs_world_avg', 'total_no', 'total_votes'
            ]
            top_opposed_output_cols_present = [col for col in top_opposed_output_cols if col in df_opposed.columns]
            top_opposed = dataframe_to_json_list(df_opposed[top_opposed_output_cols_present])
        else:
             logging.warning(f"No topics with votes > 0 found for {country_iso} in the period after country aggregation.")
    else:
        logging.warning(f"No topic vote data for {country_iso} in the period.")
        
    results['top_supported_topics'] = top_supported
    results['top_opposed_topics'] = top_opposed
    results['all_topic_voting'] = all_topic_voting

    logging.info(f"Report generation finished for {country_iso}.")
    return results

# Removed the if __name__ == "__main__": block as this is now a module. 