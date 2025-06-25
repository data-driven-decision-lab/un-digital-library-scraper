import pandas as pd
import numpy as np
import os
import glob
import sys
import logging
import argparse
from collections import Counter
import warnings
from scipy.stats import boxcox
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import ast

# ==============================================================================
# INITIAL SETUP
# ==============================================================================

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- Path Constants ---
# Assumes the script is in 'combined_dashboard_pipeline' at the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add project root to sys.path to allow for dictionary imports
sys.path.insert(0, PROJECT_ROOT)

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'pipeline_output')
DICTIONARIES_DIR = os.path.join(PROJECT_ROOT, 'dictionaries')
API_CSVS_DIR = os.path.join(PROJECT_ROOT, 'un_report_api', 'app', 'required_csvs')

# --- Dictionary Import ---
try:
    from dictionaries.un_classification import un_classification
    main_category_keys = set(un_classification.keys())
    logging.info("Successfully imported 'un_classification' dictionary.")
except ImportError:
    logging.error("Could not import 'un_classification'. Ensure 'dictionaries/un_classification.py' exists.")
    un_classification = None
    main_category_keys = set()

# ==============================================================================
# UTILITY FUNCTIONS (Consolidated)
# ==============================================================================

def find_latest_raw_data_csv(directory):
    """Finds the most recent raw voting data CSV file."""
    logging.info(f"Searching for latest raw data file in: {directory}")
    pattern = os.path.join(directory, 'UN_VOTING_DATA_RAW_WITH_TAGS_*.csv')
    list_of_files = glob.glob(pattern)
    if not list_of_files:
        logging.error(f"No CSV files found matching the pattern in '{directory}'.")
        return None
    latest_file = max(list_of_files, key=os.path.getmtime)
    logging.info(f"Found latest input file: {os.path.basename(latest_file)}")
    return latest_file

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
            final_df[f'{pillar}_Rank'] = final_df.groupby('Year')[pillar].rank(method='dense', ascending=False).astype(pd.Int64Dtype())

    if normalized_pillar_cols:
        # Change 1: Calculate 'Total Index Average' from the mean of *normalized* pillars.
        final_df['Total Index Average'] = final_df[normalized_pillar_cols].mean(axis=1, skipna=True)
        
        # The 'Total Index Normalized' is now a direct copy of this new average, without re-normalizing.
        final_df['Total Index Normalized'] = final_df['Total Index Average']

        final_df['Overall Rank'] = final_df.groupby('Year')['Total Index Average'].rank(method='dense', ascending=False).astype(pd.Int64Dtype())
        final_df.sort_values(by=['Country', 'Year'], inplace=True)
        final_df['Overall Rank Rolling Avg (3y)'] = final_df.groupby('Country')['Overall Rank'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        # Redundant normalization step has been removed.
    else:
        # Fallback for safety, though pillars should exist. This logic remains unchanged.
        raw_pillar_cols = [p for p in pillars if p in final_df.columns]
        if raw_pillar_cols:
            final_df['Total Index Average'] = final_df[raw_pillar_cols].mean(axis=1, skipna=True)
            final_df['Overall Rank'] = final_df.groupby('Year')['Total Index Average'].rank(method='dense', ascending=False).astype(pd.Int64Dtype())
            final_df.sort_values(by=['Country', 'Year'], inplace=True)
            final_df['Overall Rank Rolling Avg (3y)'] = final_df.groupby('Country')['Overall Rank'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
            final_df['Total Index Normalized'] = final_df.groupby('Year')['Total Index Average'].transform(min_max_normalize_100)

    final_df.rename(columns={'Country': 'Country name', 'Pillar1': 'Pillar 1 Score', 'Pillar2': 'Pillar 2 Score', 'Pillar3': 'Pillar 3 Score', 'Pillar1_Normalized': 'Pillar 1 Normalized', 'Pillar1_Rank': 'Pillar 1 Rank', 'Pillar2_Normalized': 'Pillar 2 Normalized', 'Pillar2_Rank': 'Pillar 2 Rank', 'Pillar3_Normalized': 'Pillar 3 Normalized', 'Pillar3_Rank': 'Pillar 3 Rank'}, inplace=True)
    
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
        if pd.isna(tag_string) or not isinstance(tag_string, str): return []
        tag_items = [item.strip() for item in tag_string.split(',') if item.strip()]
        if not tag_items: return []
        found_tags = []
        i = 0
        while i < len(tag_items):
            current_item = tag_items[i]
            if current_item in main_category_keys:
                tag_to_add = current_item
                if i + 1 < len(tag_items):
                    next_item = tag_items[i+1]
                    if next_item in un_classification.get(current_item, {}):
                        tag_to_add = next_item
                        i += 1
                found_tags.append(tag_to_add)
            i += 1
        return list(set(found_tags)) if found_tags else ["No Tag"]

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
    Orchestrates the entire data pipeline from loading raw data to saving
    the final CSVs for the API.
    """
    logging.info("Starting the Combined Dashboard Pipeline...")

    # --- 1. Load All Inputs ---
    raw_data_path = find_latest_raw_data_csv(RAW_DATA_DIR)
    if not raw_data_path:
        sys.exit(1) # Exit if no raw data is found

    region_mapping_path = os.path.join(DICTIONARIES_DIR, 'UN_Country_Region_Mapping.csv')
    region_mapping = load_region_mapping(region_mapping_path)
    # Note: The pipeline can continue without region mapping, Pillar 2 will be skipped.

    try:
        df_raw = pd.read_csv(raw_data_path, low_memory=False)
        logging.info(f"Successfully loaded raw data with shape: {df_raw.shape}")
    except Exception as e:
        logging.error(f"Fatal error loading raw data CSV: {e}")
        sys.exit(1)

    # --- 2. Initial Preprocessing (Apply to raw_df ONCE) ---
    # Filter out Security Council Resolutions
    if 'Resolution' in df_raw.columns:
        initial_count = len(df_raw)
        df_raw = df_raw[~df_raw['Resolution'].str.startswith('S/', na=False)].copy()
        logging.info(f"Filtered out Security Council votes. Kept {len(df_raw)} of {initial_count} records.")
    else:
        logging.warning("'Resolution' column not found, cannot filter out Security Council votes.")

    # Basic date and year processing
    try:
        df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
        df_raw.dropna(subset=['Date'], inplace=True)
        df_raw['Year'] = df_raw['Date'].dt.year
    except KeyError:
        logging.error("Fatal: 'Date' column not found in raw data.")
        sys.exit(1)

    # --- 3. Execute Pipeline Steps ---
    logging.info("Executing pipeline steps...")

    # Run the parallel tasks first
    df_topic_votes = generate_topic_votes(df_raw.copy()) # Pass copy to be safe
    df_similarity = generate_similarity_matrix(df_raw.copy()) # Pass copy

    # Run the sequential tasks
    df_combined_index = generate_combined_index(df_raw, region_mapping)
    df_annual_scores = generate_annual_scores(df_combined_index)

    # --- 4. Save Final Outputs ---
    os.makedirs(API_CSVS_DIR, exist_ok=True)
    logging.info(f"Saving final CSVs to: {API_CSVS_DIR}")

    # Save annual scores
    if df_annual_scores is not None and not df_annual_scores.empty:
        output_path = os.path.join(API_CSVS_DIR, 'annual_scores.csv')
        df_annual_scores.to_csv(output_path, index=False, float_format='%.4f')
        logging.info(f"Successfully saved 'annual_scores.csv' with {len(df_annual_scores)} rows.")
    else:
        logging.warning("'annual_scores.csv' was not generated or is empty.")

    # Save topic votes
    if df_topic_votes is not None and not df_topic_votes.empty:
        output_path = os.path.join(API_CSVS_DIR, 'topic_votes_yearly.csv')
        df_topic_votes.to_csv(output_path, index=False)
        logging.info(f"Successfully saved 'topic_votes_yearly.csv' with {len(df_topic_votes)} rows.")
    else:
        logging.warning("'topic_votes_yearly.csv' was not generated or is empty.")

    # Save similarity matrix
    if df_similarity is not None and not df_similarity.empty:
        output_path = os.path.join(API_CSVS_DIR, 'pairwise_similarity_yearly.csv')
        df_similarity.to_csv(output_path, index=False, float_format='%.6f')
        logging.info(f"Successfully saved 'pairwise_similarity_yearly.csv' with {len(df_similarity)} rows.")
    else:
        logging.warning("'pairwise_similarity_yearly.csv' was not generated or is empty.")

    logging.info("Combined Dashboard Pipeline finished successfully.")


if __name__ == '__main__':
    # Using argparse to allow for potential future options, like forcing a specific input file.
    parser = argparse.ArgumentParser(description="Run the full UN voting data processing pipeline.")
    # Example for a future argument:
    # parser.add_argument("-f", "--force-input-file", help="Specify a direct path to an input CSV, ignoring the latest.", default=None)
    args = parser.parse_args()

    # Suppress common numpy warnings during group operations
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')

    main() 