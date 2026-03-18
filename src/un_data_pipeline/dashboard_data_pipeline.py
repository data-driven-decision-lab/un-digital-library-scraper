import pandas as pd
import numpy as np
import os
import sys
import logging
import time
import json
import uuid
from collections import Counter
from datetime import datetime
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import ast
import libsql_experimental as libsql

# ==============================================================================
# INITIAL SETUP
# ==============================================================================

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Path Constants ---
# Set paths relative to the project root.
REFERENCE_DATA_DIR = '../../data/reference'
EXPECTED_LATEST_YEAR = 2025

# --- Dictionary Import ---
# Add the 'src' directory to sys.path to allow for package imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # This should be the project root
sys.path.insert(0, PROJECT_ROOT)

try:
    from src.un_data_pipeline.data_modules.un_classification import un_classification
    main_category_keys = set(un_classification.keys())
    subcategory_keys = set()
    for _mc_dict in un_classification.values():
        subcategory_keys.update(_mc_dict.keys())
    logging.info("Successfully imported 'un_classification' dictionary.")
except ImportError:
    logging.error("Could not import 'un_classification'. Ensure 'dictionaries/un_classification.py' exists.")
    un_classification = None
    main_category_keys = set()
    subcategory_keys = set()

# ==============================================================================
# TURSO FUNCTIONS
# ==============================================================================

def get_turso_connection():
    """Get libsql connection to Turso."""
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    if not url:
        raise ValueError("TURSO_DATABASE_URL environment variable not set.")
    if not auth_token:
        raise ValueError("TURSO_AUTH_TOKEN environment variable not set.")
    return libsql.connect(url, auth_token=auth_token)


def _expand_vote_data(df):
    """
    Expand the vote_data JSON column into per-country columns.
    Called after loading from Turso when vote_data column is present.
    """
    if 'vote_data' not in df.columns:
        return df
    try:
        vote_df = df['vote_data'].apply(json.loads).apply(pd.Series)
        df = pd.concat([df.drop('vote_data', axis=1), vote_df], axis=1)
    except Exception as e:
        logging.warning(f"Could not expand vote_data column: {e}")
    return df


def load_data_from_turso(table_name='un_votes_with_sc', page_size=1000):
    """
    Loads all data from a Turso (LibSQL) table.

    Args:
        table_name: Name of the table to load from
        page_size: Unused — kept for API compatibility; LibSQL fetches all rows at once

    Returns:
        pandas.DataFrame: Loaded data
    """
    logging.info(f"Loading data from Turso table: {table_name}")

    try:
        conn = get_turso_connection()

        # Get column names first
        cursor = conn.execute(f"SELECT * FROM {table_name} LIMIT 1")
        cols = [d[0] for d in cursor.description]

        # Fetch all rows
        rows = conn.execute(f"SELECT * FROM {table_name}").fetchall()

        if not rows:
            logging.warning(f"No data found in {table_name} table")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=cols)
        logging.info(f"Successfully loaded {len(df)} rows from {table_name}")

        # Expand vote_data JSON column if present
        df = _expand_vote_data(df)

        return df

    except Exception as e:
        logging.error(f"Error loading data from Turso table {table_name}: {e}")
        return pd.DataFrame()


def save_data_to_turso(df: pd.DataFrame, table_name: str) -> int:
    """
    Saves processed data to a Turso (LibSQL) table using upsert (INSERT OR REPLACE).

    Args:
        df: DataFrame to save
        table_name: Name of the Turso table to save to

    Returns:
        int: Number of rows saved
    """
    if df.empty:
        logging.info(f"No data to save to {table_name}")
        return 0

    logging.info(f"Saving {len(df)} rows to Turso table: {table_name}")

    try:
        conn = get_turso_connection()

        # Replace NaN with None for SQL compatibility
        df_to_upload = df.where(pd.notna(df), None)

        # Handle datetime columns — convert to ISO string
        for col in df_to_upload.columns:
            if pd.api.types.is_datetime64_any_dtype(df_to_upload[col]):
                df_to_upload[col] = df_to_upload[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Handle numeric columns: keep floats as floats (LibSQL handles them natively).
        # Keep the abs(x) > 1e3 guard for safety (Phase 3 will address this per PIPE-05).
        # Remove the old string-conversion step that was only needed for JSON/Supabase.
        for col in df_to_upload.columns:
            if col in ['Yes Votes', 'No Votes', 'Abstain Votes', 'Total Votes in Year',
                       'YesVotes_Topic', 'NoVotes_Topic', 'AbstainVotes_Topic', 'TotalVotes_Topic',
                       'Year', 'Overall Rank', 'Pillar 1 Rank', 'Pillar 2 Rank', 'Pillar 3 Rank']:
                df_to_upload[col] = pd.to_numeric(df_to_upload[col], errors='coerce').astype('Int64')
                # Convert pandas NA to Python None for libsql compatibility
                df_to_upload[col] = df_to_upload[col].where(pd.notna(df_to_upload[col]), None)
                df_to_upload[col] = df_to_upload[col].apply(lambda x: int(x) if x is not None else None)
            elif col in ['Pillar 1 Score', 'Pillar 2 Score', 'Pillar 3 Score',
                         'Total Index Average', 'Overall Rank Rolling Avg (3y)',
                         'Total Index Normalized', 'Pillar 1 Normalized', 'Pillar 2 Normalized',
                         'Pillar 3 Normalized', 'CosineSimilarity']:
                df_to_upload[col] = pd.to_numeric(df_to_upload[col], errors='coerce')
                df_to_upload[col] = df_to_upload[col].replace([np.inf, -np.inf], None)
                # Keep abs(x) > 1e3 guard (Phase 3 fix per PIPE-05)
                df_to_upload[col] = df_to_upload[col].apply(
                    lambda x: None if pd.isna(x) or (x is not None and abs(x) > 1e3) else x
                )
                df_to_upload[col] = df_to_upload[col].apply(
                    lambda x: round(x, 4) if x is not None and isinstance(x, float) else x
                )

        # Remove id column if present (auto-generated by DB)
        cols = [c for c in df_to_upload.columns if c != 'id']

        placeholders = ', '.join(['?' for _ in cols])
        col_names = ', '.join([f'"{c}"' for c in cols])
        sql = f'INSERT OR REPLACE INTO {table_name} ({col_names}) VALUES ({placeholders})'

        # Build rows as list of tuples
        all_rows = [tuple(row[c] for c in cols) for _, row in df_to_upload.iterrows()]

        # Batch in groups of 1000 to avoid memory issues
        batch_size = 1000
        total_rows = len(all_rows)
        num_batches = (total_rows + batch_size - 1) // batch_size

        logging.info(f"Upserting {total_rows} rows into {table_name} in {num_batches} batch(es)")

        for i in range(num_batches):
            batch = all_rows[i * batch_size:(i + 1) * batch_size]
            conn.executemany(sql, batch)
            conn.commit()
            logging.info(f"Upserted batch {i + 1}/{num_batches} ({len(batch)} rows)")

        logging.info(f"Successfully saved {total_rows} rows to {table_name}")
        return total_rows

    except Exception as e:
        logging.error(f"Error saving data to Turso table {table_name}: {e}")
        raise


# ==============================================================================
# UTILITY FUNCTIONS (Consolidated)
# ==============================================================================

# Removed find_latest_raw_data_csv - now using Turso

def identify_country_columns(df_columns):
    """Identifies likely country ISO3 columns (3 uppercase letters)."""
    potential_countries = [col for col in df_columns if isinstance(col, str) and len(col) == 3 and col.isupper()]
    known_non_countries = {'YES', 'NO'}
    return sorted([col for col in potential_countries if col not in known_non_countries])

def load_region_mapping(mapping_file_path):
    """Loads the country to UN region mapping CSV."""
    try:
        df_regions = pd.read_csv(mapping_file_path)
        iso_col = df_regions.columns[2].strip()
        region_col = df_regions.columns[3].strip()
        df_regions.dropna(subset=[iso_col, region_col], inplace=True)
        mapping = pd.Series(df_regions[region_col].values, index=df_regions[iso_col]).to_dict()
        logging.info(f"Loaded region mapping for {len(mapping)} countries.")
        if 'RUS' in mapping and 'USSR' not in mapping:
            mapping['USSR'] = mapping['RUS']
        return mapping
    except Exception as e:
        logging.error(f"Failed to load or process region mapping file {mapping_file_path}: {e}")
        return None

def validate_source_year_coverage(df_raw, expected_year=EXPECTED_LATEST_YEAR):
    """Validates source data includes at least the expected latest year."""
    if df_raw.empty:
        raise ValueError("Source data is empty; cannot validate year coverage.")
    if 'Date' not in df_raw.columns:
        raise ValueError("Source data missing required 'Date' column for year validation.")

    dates = pd.to_datetime(df_raw['Date'], errors='coerce')
    years = dates.dt.year.dropna()
    if years.empty:
        raise ValueError("Source data has no parseable dates; cannot validate year coverage.")

    max_year = int(years.max())
    logging.info(f"Source data year coverage: min={int(years.min())}, max={max_year}")
    if max_year < expected_year:
        raise ValueError(
            f"Source data max year is {max_year}, expected at least {expected_year}. "
            "Refusing to generate outputs from incomplete source data."
        )

def validate_output_contains_year(df, output_name, expected_year=EXPECTED_LATEST_YEAR):
    """Ensures an output dataframe contains the expected year in its Year/year column."""
    if df is None or df.empty:
        raise ValueError(f"{output_name} is empty; expected output including year {expected_year}.")

    year_col = None
    if 'Year' in df.columns:
        year_col = 'Year'
    elif 'year' in df.columns:
        year_col = 'year'

    if not year_col:
        raise ValueError(f"{output_name} has no Year/year column; cannot validate {expected_year} coverage.")

    years = pd.to_numeric(df[year_col], errors='coerce').dropna().astype(int)
    if years.empty:
        raise ValueError(f"{output_name} has no parseable year values; expected year {expected_year}.")
    if expected_year not in set(years.tolist()):
        raise ValueError(
            f"{output_name} is missing required year {expected_year}. "
            f"Available year range: {int(years.min())}-{int(years.max())}."
        )

# ==============================================================================
# DATA PROCESSING PIPELINE
#
# Each major step from the original scripts is refactored into a function that
# takes a DataFrame as input and returns a new DataFrame. This makes the
# pipeline modular and easier to debug. File I/O is handled by the main
# orchestrator.
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 1: GENERATE COMBINED INDEX
# (Logic from 'combined_index_script.py')
# ------------------------------------------------------------------------------

def generate_combined_index(df_main, country_to_region_map, bloc_size_p1=4):
    """
    Takes raw voting data and generates a DataFrame with Pillar 1, 2, and 3 scores,
    along with ranks and normalizations.
    """
    logging.info("Step 1: Starting Combined Index generation...")

    # --- Helper functions specific to this step ---
    def calculate_cosine_similarity(vec1, vec2):
        if np.isnan(vec1).any() or np.isnan(vec2).any(): return np.nan
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0: return np.nan
        return np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)

    def min_max_normalize_100(series):
        """Applies Min-Max scaling to a series to fit it into a 0-100 range."""
        min_val = series.min()
        max_val = series.max()
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            return pd.Series(50.0, index=series.index)
        return 100 * (series - min_val) / (max_val - min_val)

    def parse_tags_p1(tag_string):
        if un_classification is None or pd.isna(tag_string): return None
        try:
            tags_flat = [str(item).strip() for item in ast.literal_eval(tag_string)[0]]
        except:
            tags_flat = [tag.strip() for tag in str(tag_string).strip('[]').split(',')]

        for main_tag in tags_flat:
            if main_tag in un_classification:
                for sub_tag in tags_flat:
                    if sub_tag in un_classification.get(main_tag, {}):
                        return sub_tag
        return None

    def calculate_alignment_score_p1(df_country_bloc, bloc_years):
        num_bloc_years = len(bloc_years)
        if df_country_bloc.empty or num_bloc_years == 0: return np.nan
        tag_year_counts = df_country_bloc.groupby('tag_group')['Year'].nunique()
        consistent_tags = tag_year_counts[tag_year_counts == num_bloc_years].index
        df_filtered = df_country_bloc[df_country_bloc['tag_group'].isin(consistent_tags)]
        if df_filtered.empty: return np.nan
        total_votes_all_consistent_tags = len(df_filtered)
        all_weighted_deviations = []
        for _, group_data in df_filtered.groupby('tag_group'):
            total_votes_tag = len(group_data)
            if total_votes_tag == 0: continue
            avg_pct = {v: group_data['vote'].tolist().count(v) / total_votes_tag * 100 for v in ['YES', 'NO', 'ABSTAIN']}
            yearly_deviations = []
            for year in bloc_years:
                year_data = group_data[group_data['Year'] == year]
                total_votes_year = len(year_data)
                if total_votes_year == 0:
                    yearly_deviation_normalized = 0.0
                else:
                    year_pct = {v: year_data['vote'].tolist().count(v) / total_votes_year * 100 for v in ['YES', 'NO', 'ABSTAIN']}
                    yearly_raw_deviation = sum(abs(year_pct[v] - avg_pct[v]) for v in ['YES', 'NO', 'ABSTAIN'])
                    yearly_deviation_normalized = yearly_raw_deviation / 200.0
                yearly_deviations.append(yearly_deviation_normalized)
            weighted_deviation = np.mean(yearly_deviations) * total_votes_tag
            all_weighted_deviations.append(weighted_deviation)
        if total_votes_all_consistent_tags == 0: return np.nan
        score = max(0.0, 1.0 - (sum(all_weighted_deviations) / total_votes_all_consistent_tags))
        return score

    def run_pillar1_analysis(df_wide, country_columns, bloc_size):
        if un_classification is None:
            logging.warning("Pillar 1 skipped: un_classification dictionary not available.")
            return pd.DataFrame(columns=['Year', 'Country', 'Pillar1'])
        logging.info("... starting Pillar 1 analysis")
        df_p1 = df_wide.copy()
        df_p1['subtag1'] = df_p1['tags'].apply(parse_tags_p1)
        df_p1.dropna(subset=['subtag1'], inplace=True)
        id_vars = [col for col in ['Date', 'Year', 'Resolution', 'tags', 'subtag1'] if col in df_p1.columns]
        df_melted = df_p1.melt(id_vars=id_vars, value_vars=country_columns, var_name='Country', value_name='vote')
        df_melted = df_melted[df_melted['vote'].isin(["YES", "NO", "ABSTAIN"])].copy()
        df_melted['tag_group'] = df_melted['subtag1']
        df_melted_indexed = df_melted.set_index(['Country', 'Year']).sort_index()
        min_year, max_year = df_melted['Year'].min(), df_melted['Year'].max()
        analysis_years = range(min_year + bloc_size - 1, max_year + 1)
        p1_results = []
        for country in tqdm(country_columns, desc="Pillar 1", leave=False):
            for year_y in analysis_years:
                bloc_years = list(range(year_y - bloc_size + 1, year_y + 1))
                try:
                    bloc_data = df_melted_indexed.loc[pd.IndexSlice[country, bloc_years], :].reset_index()
                    score = calculate_alignment_score_p1(bloc_data, bloc_years)
                    if pd.notna(score):
                        p1_results.append({'Year': year_y, 'Country': country, 'Pillar1': score * 100})
                except KeyError: continue
        return pd.DataFrame(p1_results)

    def run_pillar2_analysis(df_wide, country_columns, country_to_region):
        if not country_to_region:
            logging.warning("Pillar 2 skipped: Region mapping not available.")
            return pd.DataFrame(columns=['Year', 'Country', 'Pillar2'])
        logging.info("... starting Pillar 2 analysis")
        p2_results = []
        mapped_countries = [c for c in country_columns if c in country_to_region]
        for year in tqdm(sorted(df_wide['Year'].unique()), desc="Pillar 2", leave=False):
            df_year = df_wide[df_wide['Year'] == year].copy()
            if df_year.empty: continue
            unique_regions = set(country_to_region[c] for c in mapped_countries if c in df_year.columns)
            for region in unique_regions:
                region_cols = [c for c in mapped_countries if country_to_region.get(c) == region and c in df_year.columns]
                if not region_cols: continue
                v_region_year_counts = df_year[region_cols].stack().value_counts()
                v_region_year_total = v_region_year_counts.sum()
                v_region_year = np.array([v_region_year_counts.get(v, 0) / v_region_year_total * 100 for v in ['YES', 'NO', 'ABSTAIN']]) if v_region_year_total > 0 else np.array([np.nan]*3)
                df_year[f'maj_{region}'] = df_year[region_cols].apply(lambda row: Counter(row.dropna()).most_common(1)[0][0] if Counter(row.dropna()) and Counter(row.dropna()).most_common(1)[0][1] > Counter(row.dropna()).get(Counter(row.dropna()).most_common(2)[-1][0] if len(Counter(row.dropna()).most_common(2)) > 1 else None, 0) else 'TIE', axis=1)
                for country in region_cols:
                    valid_maj = df_year[[country, f'maj_{region}']].dropna()
                    valid_maj = valid_maj[valid_maj[f'maj_{region}'] != 'TIE']
                    bmm = (valid_maj[country] == valid_maj[f'maj_{region}']).sum() / len(valid_maj) * 100 if not valid_maj.empty else np.nan
                    v_country_year_counts = df_year[country].value_counts()
                    v_country_year_total = v_country_year_counts.sum()
                    v_country_year = np.array([v_country_year_counts.get(v, 0) / v_country_year_total * 100 for v in ['YES', 'NO', 'ABSTAIN']]) if v_country_year_total > 0 else np.array([np.nan]*3)
                    cos_sim = calculate_cosine_similarity(v_country_year, v_region_year)
                    bds = cos_sim * 100 if pd.notna(cos_sim) else np.nan
                    score = np.nanmean([bmm, bds]) if not (pd.isna(bmm) or pd.isna(bds)) else np.nan
                    if pd.notna(score): p2_results.append({'Year': year, 'Country': country, 'Pillar2': score})
        return pd.DataFrame(p2_results)

    def run_pillar3_analysis(df_wide, country_columns):
        logging.info("... starting Pillar 3 analysis")
        p3_results = []
        for year in tqdm(sorted(df_wide['Year'].unique()), desc="Pillar 3", leave=False):
            df_year = df_wide[df_wide['Year'] == year].copy()
            if df_year.empty: continue
            df_year['global_majority_vote'] = df_year[country_columns].apply(lambda row: Counter(row.dropna()).most_common(1)[0][0] if Counter(row.dropna()) and Counter(row.dropna()).most_common(1)[0][1] > Counter(row.dropna()).get(Counter(row.dropna()).most_common(2)[-1][0] if len(Counter(row.dropna()).most_common(2)) > 1 else None, 0) else 'TIE', axis=1)
            v_global_year_counts = df_year[country_columns].stack().value_counts()
            v_global_year_total = v_global_year_counts.sum()
            v_global_year = np.array([v_global_year_counts.get(v, 0) / v_global_year_total * 100 for v in ['YES', 'NO', 'ABSTAIN']]) if v_global_year_total > 0 else np.array([np.nan]*3)
            for country in country_columns:
                valid_maj = df_year[[country, 'global_majority_vote']].dropna()
                valid_maj = valid_maj[valid_maj['global_majority_vote'] != 'TIE']
                gmmc = (valid_maj[country] == valid_maj['global_majority_vote']).sum() / len(valid_maj) * 100 if not valid_maj.empty else np.nan
                v_country_year_counts = df_year[country].value_counts()
                v_country_year_total = v_country_year_counts.sum()
                v_country_year = np.array([v_country_year_counts.get(v, 0) / v_country_year_total * 100 for v in ['YES', 'NO', 'ABSTAIN']]) if v_country_year_total > 0 else np.array([np.nan]*3)
                cos_sim = calculate_cosine_similarity(v_country_year, v_global_year)
                gdsc = cos_sim * 100 if pd.notna(cos_sim) else np.nan
                score = np.nanmean([gmmc, gdsc]) if not (pd.isna(gmmc) or pd.isna(gdsc)) else np.nan
                if pd.notna(score): p3_results.append({'Year': year, 'Country': country, 'Pillar3': score})
        return pd.DataFrame(p3_results)

    country_columns = identify_country_columns(df_main.columns)
    if not country_columns:
        logging.error("COMBINED INDEX: No country columns identified. Aborting this step.")
        return pd.DataFrame()

    logging.info("COMBINED INDEX: Calculating vote counts...")
    df_melted_counts = df_main.melt(id_vars=['Year'], value_vars=country_columns, var_name='Country', value_name='vote')
    counts_grouped = df_melted_counts.groupby(['Year', 'Country', 'vote']).size().unstack(fill_value=0)
    for col in ['YES', 'NO', 'ABSTAIN']:
        if col not in counts_grouped.columns: counts_grouped[col] = 0
    counts_grouped.rename(columns={'YES': 'Yes Votes', 'NO': 'No Votes', 'ABSTAIN': 'Abstain Votes'}, inplace=True)
    counts_grouped['Total Votes in Year'] = counts_grouped.sum(axis=1)
    df_vote_counts = counts_grouped.reset_index()

    df_p1 = run_pillar1_analysis(df_main, country_columns, bloc_size_p1)
    df_p2 = run_pillar2_analysis(df_main, country_columns, country_to_region_map)
    df_p3 = run_pillar3_analysis(df_main, country_columns)

    logging.info("COMBINED INDEX: Combining pillar results...")
    final_df = df_vote_counts
    for df_p, name in [(df_p1, 'Pillar1'), (df_p2, 'Pillar2'), (df_p3, 'Pillar3')]:
        if not df_p.empty:
            final_df = pd.merge(final_df, df_p, on=['Year', 'Country'], how='left')
        else: final_df[name] = np.nan

    logging.info("COMBINED INDEX: Normalizing and ranking...")
    pillars = ['Pillar1', 'Pillar2', 'Pillar3']
    normalized_pillar_cols = []
    for pillar in pillars:
        if pillar in final_df.columns:
            normalized_col_name = f'{pillar}_Normalized'
            final_df[normalized_col_name] = final_df.groupby('Year')[pillar].transform(min_max_normalize_100)
            normalized_pillar_cols.append(normalized_col_name)
            final_df[f'{pillar}_Rank'] = final_df.groupby('Year')[pillar].rank(method='min', ascending=False).astype(pd.Int64Dtype())

    if normalized_pillar_cols:
        # Change 1: Calculate 'Total Index Average' from the mean of *normalized* pillars.
        final_df['Total Index Average'] = final_df[normalized_pillar_cols].mean(axis=1, skipna=True)

        # The 'Total Index Normalized' is now a direct copy of this new average, without re-normalizing.
        final_df['Total Index Normalized'] = final_df['Total Index Average']

        final_df['Overall Rank'] = final_df.groupby('Year')['Total Index Average'].rank(method='min', ascending=False).astype(pd.Int64Dtype())
        final_df.sort_values(by=['Country', 'Year'], inplace=True)
        final_df['Overall Rank Rolling Avg (3y)'] = final_df.groupby('Country')['Overall Rank'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        # Redundant normalization step has been removed.
    else:
        # Fallback for safety, though pillars should exist. This logic remains unchanged.
        raw_pillar_cols = [p for p in pillars if p in final_df.columns]
        if raw_pillar_cols:
            final_df['Total Index Average'] = final_df[raw_pillar_cols].mean(axis=1, skipna=True)
            final_df['Overall Rank'] = final_df.groupby('Year')['Total Index Average'].rank(method='min', ascending=False).astype(pd.Int64Dtype())
            final_df.sort_values(by=['Country', 'Year'], inplace=True)
            final_df['Overall Rank Rolling Avg (3y)'] = final_df.groupby('Country')['Overall Rank'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            final_df['Total Index Normalized'] = final_df.groupby('Year')['Total Index Average'].transform(min_max_normalize_100)

    final_df.rename(columns={'Country': 'Country name', 'Pillar1': 'Pillar 1 Score', 'Pillar2': 'Pillar 2 Score', 'Pillar3': 'Pillar 3 Score', 'Pillar1_Normalized': 'Pillar 1 Normalized', 'Pillar1_Rank': 'Pillar 1 Rank', 'Pillar2_Normalized': 'Pillar 2 Normalized', 'Pillar2_Rank': 'Pillar 2 Rank', 'Pillar3_Normalized': 'Pillar 3 Normalized', 'Pillar3_Rank': 'Pillar 3 Rank'}, inplace=True)

    # Convert 2-letter country codes to 3-letter codes
    logging.info("Converting country codes from 2-letter to 3-letter format...")
    try:
        import pycountry
        def convert_country_code(country_code):
            if len(country_code) == 2:
                try:
                    country = pycountry.countries.get(alpha_2=country_code)
                    return country.alpha_3 if country else country_code
                except:
                    return country_code
            return country_code

        final_df['Country name'] = final_df['Country name'].apply(convert_country_code)
        logging.info("Country code conversion completed.")
    except ImportError:
        logging.warning("pycountry not available, skipping country code conversion.")
    except Exception as e:
        logging.warning(f"Error converting country codes: {e}")

    logging.info("Step 1: Combined Index generation finished.")
    return final_df

# ------------------------------------------------------------------------------
# STEP 2: GENERATE ANNUAL SCORES
# (Logic from 'annual_pillar_breakdown.py')
# ------------------------------------------------------------------------------

def generate_annual_scores(df_combined_index):
    """
    Processes the output from the combined index step to create the final
    'annual_scores.csv' data.
    """
    if df_combined_index is None or df_combined_index.empty:
        logging.warning("Annual Scores step skipped: Input DataFrame is empty.")
        return pd.DataFrame()

    logging.info("Step 2: Starting Annual Scores generation...")

    # Identify country column
    country_col = 'Country name' if 'Country name' in df_combined_index.columns else 'Country'
    if country_col not in df_combined_index.columns:
        logging.error("ANNUAL SCORES: Cannot find country column. Aborting this step.")
        return pd.DataFrame()

    # Define all possible columns we want in the final output
    core_cols = [
        country_col, 'Year',
        'Pillar 1 Score', 'Pillar 2 Score', 'Pillar 3 Score',
        'Total Index Average', 'Overall Rank', 'Overall Rank Rolling Avg (3y)',
        'Total Index Normalized', 'Pillar 1 Normalized', 'Pillar 1 Rank',
        'Pillar 2 Normalized', 'Pillar 2 Rank', 'Pillar 3 Normalized', 'Pillar 3 Rank',
        'Yes Votes', 'No Votes', 'Abstain Votes', 'Total Votes in Year'
    ]

    # Filter to only columns that actually exist in the input dataframe
    cols_to_keep = [col for col in core_cols if col in df_combined_index.columns]

    if len(cols_to_keep) <= 2: # Only country and year found
        logging.error("ANNUAL SCORES: No relevant score/rank or vote columns found. Aborting.")
        return pd.DataFrame()

    df_annual = df_combined_index[cols_to_keep].copy()

    # Change 2: Overwrite 'score' columns with their 'normalized' counterparts for the final output.
    logging.info("ANNUAL SCORES: Overwriting score columns with normalized values for export.")
    if 'Pillar 1 Normalized' in df_annual.columns and 'Pillar 1 Score' in df_annual.columns:
        df_annual['Pillar 1 Score'] = df_annual['Pillar 1 Normalized']
    if 'Pillar 2 Normalized' in df_annual.columns and 'Pillar 2 Score' in df_annual.columns:
        df_annual['Pillar 2 Score'] = df_annual['Pillar 2 Normalized']
    if 'Pillar 3 Normalized' in df_annual.columns and 'Pillar 3 Score' in df_annual.columns:
        df_annual['Pillar 3 Score'] = df_annual['Pillar 3 Normalized']
    if 'Total Index Normalized' in df_annual.columns and 'Total Index Average' in df_annual.columns:
        df_annual['Total Index Average'] = df_annual['Total Index Normalized']

    # Ensure numeric types for all score/vote/rank columns for consistency
    numeric_cols = [col for col in cols_to_keep if col not in [country_col, 'Year']]
    for col in numeric_cols:
        df_annual[col] = pd.to_numeric(df_annual[col], errors='coerce')

    logging.info(f"Step 2: Annual Scores generation finished. Shape: {df_annual.shape}")
    return df_annual

# ------------------------------------------------------------------------------
# STEP 3A: GENERATE TOPIC VOTES
# (Logic from 'aggregate_topic_votes.py')
# ------------------------------------------------------------------------------

def generate_topic_votes(df_raw):
    """
    Aggregates votes yearly by country and topic (subtag1).
    """
    if un_classification is None:
        logging.warning("Topic Votes step skipped: 'un_classification' dictionary not available.")
        return pd.DataFrame()

    logging.info("Step 3A: Starting Topic Votes generation...")

    def parse_tags_for_subtag1(tag_string):
        if pd.isna(tag_string) or not isinstance(tag_string, str):
            return []
        tag_items = [item.strip() for item in tag_string.split(',') if item.strip()]
        matched = []
        for item in tag_items:
            if item in main_category_keys or item in subcategory_keys:
                matched.append(item)
        return list(dict.fromkeys(matched))  # dedupe preserving insertion order

    country_cols = identify_country_columns(df_raw.columns)
    if not country_cols:
        logging.error("TOPIC VOTES: No country columns identified. Aborting this step.")
        return pd.DataFrame()

    if 'tags' not in df_raw.columns:
        logging.error("TOPIC VOTES: 'tags' column not found. Aborting this step.")
        return pd.DataFrame()

    logging.info("... melting dataframe for topic analysis")
    id_vars = [col for col in ['Year', 'Resolution', 'tags'] if col in df_raw.columns]
    df_melted = df_raw.melt(id_vars=id_vars, value_vars=country_cols, var_name='Country', value_name='Vote')
    df_melted = df_melted[df_melted['Vote'].isin(['YES', 'NO', 'ABSTAIN'])]
    if df_melted.empty:
        logging.warning("TOPIC VOTES: No valid votes found after melting.")
        return pd.DataFrame()

    logging.info("... parsing tags and exploding dataframe")
    df_melted['tags'] = df_melted['tags'].astype(str)
    tqdm.pandas(desc="Parsing Topic Tags", leave=False)
    df_melted['TopicTags'] = df_melted['tags'].progress_apply(parse_tags_for_subtag1)

    df_exploded = df_melted.explode('TopicTags')
    df_exploded.dropna(subset=['TopicTags'], inplace=True)
    if df_exploded.empty:
        logging.warning("TOPIC VOTES: Dataframe is empty after exploding tags. No topics found.")
        return pd.DataFrame()

    logging.info("... grouping and counting votes by topic")
    df_counts = df_exploded.groupby(['Year', 'Country', 'TopicTags', 'Vote']).size().unstack(fill_value=0)
    for vote_type in ['YES', 'NO', 'ABSTAIN']:
        if vote_type not in df_counts.columns:
            df_counts[vote_type] = 0
    df_counts = df_counts.rename(columns={'YES': 'YesVotes_Topic', 'NO': 'NoVotes_Topic', 'ABSTAIN': 'AbstainVotes_Topic'})
    df_counts['TotalVotes_Topic'] = df_counts[['YesVotes_Topic', 'NoVotes_Topic', 'AbstainVotes_Topic']].sum(axis=1)
    df_final = df_counts.reset_index().rename(columns={'TopicTags': 'TopicTag'})

    final_cols_order = ['Year', 'Country', 'TopicTag', 'YesVotes_Topic', 'NoVotes_Topic', 'AbstainVotes_Topic', 'TotalVotes_Topic']
    df_final = df_final[final_cols_order]
    df_final.drop_duplicates(subset=['Year', 'Country', 'TopicTag'], inplace=True)

    logging.info(f"Step 3A: Topic Votes generation finished. Shape: {df_final.shape}")
    return df_final

# ------------------------------------------------------------------------------
# STEP 3B: GENERATE SIMILARITY MATRIX
# (Logic from 'calculate_similarity_yearly.py')
# ------------------------------------------------------------------------------

def generate_similarity_matrix(df_raw):
    """
    Calculates pairwise cosine similarity between countries for each year.
    """
    logging.info("Step 3B: Starting Pairwise Similarity generation...")

    def map_vote(vote):
        if pd.isna(vote): return 0
        vote_str = str(vote).upper().strip()
        if vote_str == 'YES': return 1
        if vote_str == 'NO': return -1
        return 0

    country_cols = identify_country_columns(df_raw.columns)
    if not country_cols:
        logging.error("SIMILARITY: No country columns identified. Aborting this step.")
        return pd.DataFrame()

    all_year_similarities = []
    unique_years = sorted(df_raw['Year'].unique())

    for year in tqdm(unique_years, desc="Similarity per Year", leave=False):
        df_year = df_raw[df_raw['Year'] == year][country_cols]
        if df_year.empty: continue

        vote_matrix_numeric = df_year.apply(lambda col: col.map(map_vote)).fillna(0).astype(np.int8)
        if vote_matrix_numeric.empty: continue

        try:
            similarity_matrix = cosine_similarity(vote_matrix_numeric.T)
            df_sim = pd.DataFrame(similarity_matrix, index=country_cols, columns=country_cols)
            df_sim_long = df_sim.stack().reset_index()
            df_sim_long.columns = ['Country1_ISO3', 'Country2_ISO3', 'CosineSimilarity']
            df_sim_long['Year'] = year
            df_sim_long = df_sim_long[df_sim_long['Country1_ISO3'] < df_sim_long['Country2_ISO3']]
            all_year_similarities.append(df_sim_long)
        except Exception as e:
            logging.error(f"SIMILARITY: Error during calculation for year {year}: {e}")
            continue

    if not all_year_similarities:
        logging.warning("SIMILARITY: No results generated.")
        return pd.DataFrame()

    final_df = pd.concat(all_year_similarities, ignore_index=True)
    final_df = final_df[['Year', 'Country1_ISO3', 'Country2_ISO3', 'CosineSimilarity']]

    logging.info(f"Step 3B: Pairwise Similarity generation finished. Shape: {final_df.shape}")
    return final_df

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    """
    Main function to orchestrate the entire data processing pipeline.
    Now Turso-native: reads from un_votes_with_sc and saves processed data to Turso tables.
    """
    logging.info("==============================================================================")
    logging.info("Starting Turso-native Dashboard Data Pipeline")
    logging.info("==============================================================================")

    run_id = str(uuid.uuid4())

    # --- Record pipeline start in pipeline_runs ---
    conn = get_turso_connection()
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_runs (run_id, pipeline_name, started_at, status) VALUES (?, ?, ?, ?)",
        (run_id, 'dashboard_data_pipeline', datetime.utcnow().isoformat(), 'running')
    )
    conn.commit()
    logging.info(f"Recorded pipeline run start: run_id={run_id}")

    try:
        # --- 1. Load Data from Turso ---
        source_table = os.getenv('PIPELINE_SOURCE_TABLE', 'un_votes_with_sc')
        logging.info(f"Using Turso source table: {source_table}")
        df_raw = load_data_from_turso(source_table)
        if df_raw.empty:
            logging.error(f"No data found in {source_table} table. Exiting.")
            conn.execute(
                "UPDATE pipeline_runs SET finished_at=?, status=?, error_message=? WHERE run_id=?",
                (datetime.utcnow().isoformat(), 'failed', f'No data found in {source_table}', run_id)
            )
            conn.commit()
            sys.exit(1)

        logging.info(f"Successfully loaded {len(df_raw)} rows from {source_table} table.")
        validate_source_year_coverage(df_raw, expected_year=EXPECTED_LATEST_YEAR)

        # Filter out Security Council resolutions
        df_filtered = df_raw[~df_raw['Resolution'].str.startswith('S/', na=False)].copy()
        logging.info(f"Filtered out Security Council resolutions, {len(df_filtered)} rows remaining.")

        # Create 'Year' column from 'Date'
        df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], errors='coerce')
        df_filtered.dropna(subset=['Date'], inplace=True)
        df_filtered['Year'] = df_filtered['Date'].dt.year
        logging.info("Created 'Year' column from 'Date'.")

        # Load region mapping
        region_mapping_path = os.path.join(REFERENCE_DATA_DIR, 'UN_Country_Region_Mapping.csv')
        # Fix path if running from different directory
        if not os.path.exists(region_mapping_path):
            # Try relative to current working directory
            region_mapping_path = 'data/reference/UN_Country_Region_Mapping.csv'
        country_to_region_map = load_region_mapping(region_mapping_path)
        if not country_to_region_map:
            logging.warning("Continuing without region mapping. Pillar 2 will be affected.")

        # --- 2. Run Processing Steps ---
        df_combined_index = generate_combined_index(df_filtered.copy(), country_to_region_map)
        df_annual_scores = generate_annual_scores(df_combined_index.copy())
        df_topic_votes = generate_topic_votes(df_filtered.copy())
        df_similarity = generate_similarity_matrix(df_filtered.copy())

        # --- 3. Save Outputs to Turso ---
        logging.info("Saving processed data to Turso...")

        # Rename 'Country name' -> 'Country' to match Turso schema for annual_scores
        df_annual_scores_db = df_annual_scores.copy()
        if 'Country name' in df_annual_scores_db.columns:
            df_annual_scores_db = df_annual_scores_db.rename(columns={'Country name': 'Country'})

        # Rename Country1_ISO3/Country2_ISO3 -> Country1/Country2 to match Turso schema
        df_similarity_db = df_similarity.copy()
        if 'Country1_ISO3' in df_similarity_db.columns:
            df_similarity_db = df_similarity_db.rename(columns={
                'Country1_ISO3': 'Country1',
                'Country2_ISO3': 'Country2'
            })

        logging.info(f"About to save annual_scores: {df_annual_scores_db.shape}")
        logging.info(f"Annual scores columns: {list(df_annual_scores_db.columns)}")
        rows_annual = save_data_to_turso(df_annual_scores_db, 'annual_scores')

        logging.info(f"About to save topic_votes_yearly: {df_topic_votes.shape}")
        rows_topic = save_data_to_turso(df_topic_votes, 'topic_votes_yearly')

        logging.info(f"About to save pairwise_similarity_yearly: {df_similarity_db.shape}")
        rows_similarity = save_data_to_turso(df_similarity_db, 'pairwise_similarity_yearly')

        # --- 4. Save locally as CSV files (for API fallback) ---
        OUTPUT_DATA_DIR = os.path.join(PROJECT_ROOT, 'src', 'un_report_api', 'app', 'required_csvs')

        if not os.path.exists(OUTPUT_DATA_DIR):
            os.makedirs(OUTPUT_DATA_DIR)
            logging.info(f"Created output directory: {OUTPUT_DATA_DIR}")

        # Define output paths
        annual_scores_path = os.path.join(OUTPUT_DATA_DIR, 'annual_scores.csv')
        topic_votes_path = os.path.join(OUTPUT_DATA_DIR, 'topic_votes_yearly.csv')
        similarity_path = os.path.join(OUTPUT_DATA_DIR, 'pairwise_similarity_yearly.csv')

        # Save files locally
        logging.info("Saving CSV files locally...")
        df_annual_scores.to_csv(annual_scores_path, index=False)
        logging.info(f"Successfully saved annual scores to: {annual_scores_path}")

        df_topic_votes.to_csv(topic_votes_path, index=False)
        logging.info(f"Successfully saved topic votes to: {topic_votes_path}")

        df_similarity.to_csv(similarity_path, index=False)
        logging.info(f"Successfully saved similarity matrix to: {similarity_path}")

        # --- 5. Validate required year coverage in outputs ---
        validate_output_contains_year(df_annual_scores, 'annual_scores.csv', expected_year=EXPECTED_LATEST_YEAR)
        validate_output_contains_year(df_topic_votes, 'topic_votes_yearly.csv', expected_year=EXPECTED_LATEST_YEAR)
        validate_output_contains_year(df_similarity, 'pairwise_similarity_yearly.csv', expected_year=EXPECTED_LATEST_YEAR)
        logging.info(f"Validated required year coverage ({EXPECTED_LATEST_YEAR}) in all output CSV datasets.")

        # --- 6. Update pipeline_runs with success ---
        total_rows = rows_annual + rows_topic + rows_similarity
        conn.execute(
            "UPDATE pipeline_runs SET finished_at=?, status=?, rows_affected=? WHERE run_id=?",
            (datetime.utcnow().isoformat(), 'success', total_rows, run_id)
        )
        conn.commit()
        logging.info(f"Updated pipeline run: run_id={run_id}, status=success, rows_affected={total_rows}")

        logging.info("==============================================================================")
        logging.info("Pipeline finished successfully!")
        logging.info("Data saved to Turso tables: annual_scores, topic_votes_yearly, pairwise_similarity_yearly")
        logging.info("==============================================================================")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        try:
            conn.execute(
                "UPDATE pipeline_runs SET finished_at=?, status=?, error_message=? WHERE run_id=?",
                (datetime.utcnow().isoformat(), 'failed', str(e), run_id)
            )
            conn.commit()
        except Exception as update_err:
            logging.error(f"Failed to update pipeline_runs with error status: {update_err}")
        raise


if __name__ == '__main__':
    main()
