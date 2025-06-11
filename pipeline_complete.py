#!/usr/bin/env python3
"""
Combined UN Resolution Scraper & Tagging Pipeline with Enhanced Geo-Tagging

This script:
  • Connects to Supabase to fetch existing resolution links.
  • Runs the scraper to obtain new resolution rows (skipping rows already present).
  • Performs dual classification on new rows:
    - Regular tagging for subject categorization (tags column)
    - Enhanced geo-tagging for country, subregion, and continent.
  • Appends the new, tagged rows to the 'un_votes_raw' table in Supabase.
  • Writes out a local CSV file with the new data for backup.
  
Requirements:
  - Supabase environment variables (URL and Key) must be set.
  - The scraper identifies new rows based on the unique 'Link' attribute.
  - Only new rows are processed and uploaded.
"""
import sys
import os
import re
import ast
import csv
import time
import random
import logging
import platform
import json
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple, Callable
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from queue import Queue
from dictionaries.un_classification import un_classification
from dictionaries.un_geo_hierarchy import geo_hierarchy
from dictionaries.iso2_country import iso2_country_code
import pycountry
from supabase import create_client, Client

import random
import logging
os.environ['WDM_LOG_LEVEL'] = '0'
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse



# ---------------- Selenium & Scraper Imports ----------------
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementNotInteractableException,
    StaleElementReferenceException
)
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# ---------------- Configuration & Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%S",
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler("logs/un_scraper_tagger.log", mode='w')  # File handler
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set your API key
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables.")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Key not found in environment variables.")

# ---------------- Global Settings ----------------
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_WORKERS = 2  # For parallel scraping

# Create directories
os.makedirs("pipeline_output", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def create_supabase_client() -> Client:
    """Initializes and returns a Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def get_existing_links_from_supabase(client: Client) -> set:
    """Fetches all unique 'Link' values from the 'un_votes_raw' table."""
    logger.info("Fetching existing resolution links from Supabase...")
    try:
        response = client.table('un_votes_raw').select('Link').execute()
        if response.data:
            links = {item['Link'] for item in response.data if 'Link' in item}
            logger.info(f"Found {len(links)} existing links in Supabase.")
            return links
    except Exception as e:
        logger.error(f"Error fetching links from Supabase: {e}")
        return set()

# Find the most recent CSV file in the pipeline_output folder
def get_latest_master_csv():
    # Look for all CSV files with the pattern UN_VOTING_DATA_RAW_WITH_TAGS_*.csv
    pattern = "pipeline_output/UN_VOTING_DATA_RAW_WITH_TAGS_*.csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        # No existing files, create a default filename with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        return f"pipeline_output/UN_VOTING_DATA_RAW_WITH_TAGS_{today}.csv"
    
    # Option 2: Sort by date in the filename (more reliable)
    date_pattern = re.compile(r'UN_VOTING_DATA_RAW_WITH_TAGS_(\d{4}-\d{2}-\d{2})\.csv')
    
    # Extract dates from filenames and sort
    dated_files = []
    for file in csv_files:
        match = date_pattern.search(file)
        if match:
            date_str = match.group(1)
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                dated_files.append((file, file_date))
            except ValueError:
                # Skip files with invalid dates
                continue
    
    # If we found files with valid dates, get the most recent one
    if dated_files:
        latest_file = max(dated_files, key=lambda x: x[1])[0]
        return latest_file
    
    # Fallback: if date parsing fails, use modification time, or create new if none exist
    if csv_files:
        latest_file = max(csv_files, key=os.path.getmtime)
        return latest_file
    else:
        # No existing files, create a default filename with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        return f"pipeline_output/UN_VOTING_DATA_RAW_WITH_TAGS_{today}.csv"


# Set the master CSV file
MASTER_CSV = get_latest_master_csv()
print(f"Using master CSV: {MASTER_CSV}")

FIXED_COLUMNS = [
    "Council", "Date", "Title", "Resolution", "TOTAL VOTES", "NO-VOTE COUNT",
    "ABSTAIN COUNT", "NO COUNT", "YES COUNT", "Link", "token", "Scrape_Year"
]

# Scraper constants
BASE_SEARCH_URL = ("https://digitallibrary.un.org/search?cc=Voting%20Data&ln=en&p=&f=&rm=&sf=&so=d"
                   "&rg=50&c=Voting%20Data&c=&of=hb&fti=1&fct__9=Vote&fti=1")
MAX_PAGES_PER_YEAR = 50
MAX_WORKERS = 2

# User agent rotation for Selenium
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.5481.100 Safari/537.36"
]
user_agent_index = 0

def reset_user_agent_rotation():
    global user_agent_index
    user_agent_index = 0

# Custom exceptions
class DuplicateLinkFound(Exception):
    def __init__(self, message, new_links):
        super().__init__(message)
        self.new_links = new_links


# -------------------- Geo-Tagging Functions --------------------

class LocationClassifications(BaseModel):
    continent: Optional[str] = Field(None, description="Continent of the country")
    subregion: Optional[str] = Field(None, description="Subregion of the country")
    country: Optional[str] = Field(None, description="Country")

class ResolutionTarget(BaseModel):
    classifications: List[LocationClassifications] = Field(..., description="List of relevant classifications for this resolution")

def create_openai_client() -> OpenAI:
    """Create and return an OpenAI client instance using the API key."""
    return OpenAI(api_key=API_KEY)

def execute_api_call(api_call_fn, max_retries=5):
    """
    Execute an API call with robust rate limit handling.
    Uses exponential backoff with jitter on rate limit errors.

    Args:
        api_call_fn: Function that performs the API call.
        max_retries: Maximum retries before giving up.

    Returns:
        API response if successful, otherwise raises an exception.
    """
    retries = 0
    client = create_openai_client()

    while retries < max_retries:
        try:
            return api_call_fn(client)  # Attempt API call
        except Exception as e:
            # Check for rate limit error
            if hasattr(e, "response") and getattr(e.response, "status_code", None) == 429 or "429" in str(e):
                logger.warning(f"API rate limit error. Retrying... (Attempt {retries+1}/{max_retries})")
                
                # Exponential backoff with jitter
                wait_time = (2 ** retries) + random.uniform(0, 3)
                logger.warning(f"Waiting {wait_time:.2f} seconds before retrying...")
                time.sleep(wait_time)
                
                retries += 1
                continue
            else:
                raise e  # Re-raise other errors

    logger.error("Max retries reached. Halting API calls for this request.")
    raise Exception("OpenAI API request failed after max retries")

def call_llm_api(title: str, geo_hierarchy: dict, model: str = DEFAULT_MODEL) -> ResolutionTarget:
    """
    Analyzes a UN resolution text using LLM to identify if it's related to specific countries or regions.
    
    Args:
        title: Title of the resolution to analyze
        geo_hierarchy: Dictionary containing geographical hierarchy information
        model: OpenAI model to use
       
    Returns:
        ResolutionTarget: Structured classification results
    """
    # Prepare the system prompt
    system_prompt = f"""You are a UN document classification assistant. Your task is to analyze UN resolutions given their Title, which contains the name and some details.
If the resolution text does not target any specific continent (i.e., none of Africa, Americas, Antarctica, Asia, Europe, Oceania are mentioned), then return an empty string for the continent, subregion, and country.
If a resolution does target a location:
    - When it refers to a continent, include the continent and (if mentioned) the subregion, if not return empty string; set the country to empty string.
    - When it refers to a subregion, include the continent and subregion; set the country to empty string.
    - When it refers to a country, include the continent, subregion, and country.
    - For countries you should use the ISO 3166-1 alpha-2 code if available, otherwise use the country name.
    
For each resolution text, identify ALL relevant tags that apply and return them as a list.
{geo_hierarchy}
"""

    # Call the API with structured output
    api_call = lambda client: client.beta.chat.completions.parse(
        model=model,
        temperature=DEFAULT_TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Resolution text: {title}"}
        ],
        max_tokens=DEFAULT_MAX_TOKENS,
        response_format=ResolutionTarget,
    )
    
    try:
        logger.info(f"Calling LLM API for resolution classification: {title[:50]}...")
        response = execute_api_call(api_call)
        
        # Extract the parsed result
        classification_result: ResolutionTarget = response.choices[0].message.parsed
        logger.info("API call successful.")
        return classification_result
        
    except Exception as e:
        logger.error(f"Error during API call: {e}")
        # Return empty classification with error message
        return ResolutionTarget(
            classifications=[
                LocationClassifications(
                    continent="error",
                    subregion="processing_error",
                    country=None
                )
            ]
        )

def get_llm_location_tags(title: str, geo_hierarchy: dict, model: str = DEFAULT_MODEL) -> List[List]:
    """
    Calls the LLM API to get classification tags and returns a list of location details.
    
    Args:
        title: Title of the resolution
        geo_hierarchy: Dictionary containing geographical hierarchy
        model: OpenAI model to use
        
    Returns:
        List of lists containing [continent, subregion, country] 
        for each classification.
    """
    logger.info(f"Getting LLM tags for: {title[:50]}...")
    classification_result = call_llm_api(title, geo_hierarchy, model)
    
    result = []
    for classification in classification_result.classifications:
        # Only add entries that have at least a continent
        if classification.continent and classification.continent != "error":
            result.append([
                classification.continent,
                classification.subregion or "",
                classification.country or ""
            ])
    
    logger.info(f"Extracted LLM tags: {result}")
    return result

def combined_geo_tagger(df, geo_hierarchy, iso2_country_code, 
                        model: str = DEFAULT_MODEL,
                        max_workers: int = DEFAULT_MAX_WORKERS,
                        use_llm: bool = True):
    """
    Tags countries, subregions, and continents in a dataframe using both pattern matching and LLM.
    
    Steps:
    1. First uses regex pattern matching to identify explicit mentions
    2. Then uses LLM to identify any additional or implicit geographic entities
    3. Combines both results and formats according to standards
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'Title' column to analyze
    geo_hierarchy : dict
        Dictionary of geographical hierarchy information
    iso2_country_code : dict
        Dictionary mapping country names to ISO codes
    model : str
        OpenAI model to use for LLM analysis
    max_workers : int
        Maximum number of parallel worker threads (for processing rows)
    use_llm : bool
        Whether to use LLM for enhanced detection
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added 'country', 'subregion', and 'continent' columns
    """
    # Print the columns to debug
    logger.info(f"DataFrame columns before geo-tagging: {df.columns.tolist()}")
    
    # Check for and remove any existing geographic columns
    for col in ['country', 'subregion', 'continent']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
        
    logger.info(f"DataFrame columns after cleanup: {df.columns.tolist()}")
    
    # Determine insertion position (after "Resolution")
    if 'Resolution' in df.columns:
        res_idx = list(df.columns).index('Resolution')
        insert_pos = res_idx + 1
    else:
        # Default to position 5 if Resolution not found
        insert_pos = 5 if len(df.columns) > 5 else len(df.columns)
    
    # Create empty lists for our new columns
    countries_list = [[] for _ in range(len(df))]
    subregions_list = [[] for _ in range(len(df))]
    continents_list = [[] for _ in range(len(df))]
    
    # Pre-process country patterns for more accurate matching
    # Create a map of all countries and their details
    all_countries = []
    all_subregions = []
    all_continents = []
    
    for continent in geo_hierarchy.keys():
        # Add continent itself to patterns for direct continent mentions
        all_continents.append(continent)
        
        for subregion, countries_or_dict in geo_hierarchy[continent].items():
            # Add subregion itself to patterns for direct subregion mentions
            all_subregions.append((subregion, continent))
            
            if isinstance(countries_or_dict, dict):
                for subsubregion, countries in countries_or_dict.items():
                    # Add subsubregion to patterns
                    all_subregions.append((subsubregion, continent))
                    
                    for country in countries:
                        all_countries.append((country, subsubregion, continent))
            else:
                for country in countries_or_dict:
                    all_countries.append((country, subregion, continent))
    
    # Sort countries by length (descending) to avoid partial matches
    all_countries.sort(key=lambda x: len(x[0]), reverse=True)
    all_subregions.sort(key=lambda x: len(x[0]), reverse=True)
    all_continents.sort(key=len, reverse=True)
    
    # Compile regex patterns
    country_patterns = {country: re.compile(r'\b' + re.escape(country) + r'\b') 
                       for country, _, _ in all_countries}
    subregion_patterns = {subregion: re.compile(r'\b' + re.escape(subregion) + r'\b')
                         for subregion, _ in all_subregions}
    continent_patterns = {continent: re.compile(r'\b' + re.escape(continent) + r'\b')
                         for continent in all_continents}
    
    # STEP 1: Process each row with regex pattern matching
    logger.info("Starting regex pattern matching phase...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Pattern matching phase"):
        title = str(row['Title'])  # Ensure it's a string
        
        # Find all country matches
        found_countries = False
        for country, subregion, continent in all_countries:
            if country_patterns[country].search(title):
                found_countries = True
                # Get ISO code
                iso_code = iso2_country_code.get(country, country)
                if iso_code and iso_code not in countries_list[idx]:
                    countries_list[idx].append(iso_code)
                    if subregion not in subregions_list[idx]:
                        subregions_list[idx].append(subregion)
                    if continent not in continents_list[idx]:
                        continents_list[idx].append(continent)
        
        # Find all subregion matches (if not already handled by country)
        if not found_countries:  # Only check if no countries found
            found_subregions = False
            for subregion, continent in all_subregions:
                if subregion_patterns[subregion].search(title):
                    found_subregions = True
                    if subregion not in subregions_list[idx]:
                        subregions_list[idx].append(subregion)
                    if continent not in continents_list[idx]:
                        continents_list[idx].append(continent)
            
            # Find all continent matches (if no countries or subregions found)
            if not found_subregions:
                for continent in all_continents:
                    if continent_patterns[continent].search(title):
                        if continent not in continents_list[idx]:
                            continents_list[idx].append(continent)
    
    # STEP 2: Use LLM to augment results if enabled
    if use_llm:
        logger.info("Starting LLM enrichment phase...")
        
        # Function to process a single row with LLM
        def process_row_with_llm(idx):
            title = str(df.iloc[idx]['Title'])
            llm_results = get_llm_location_tags(title, geo_hierarchy, model)
            return idx, llm_results
        
        # Process in parallel if max_workers > 1
        if max_workers > 1:
            logger.info(f"Processing with {max_workers} parallel workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all rows for processing
                future_to_idx = {}
                for idx in range(len(df)):
                    future = executor.submit(process_row_with_llm, idx)
                    future_to_idx[future] = idx
                
                # Process results as they complete
                for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="LLM processing"):
                    idx, llm_results = future.result()
                    
                    # Add any new results from LLM, avoiding duplicates
                    for continent, subregion, country in llm_results:
                        if country:
                            # Convert to ISO code if possible
                            iso_code = iso2_country_code.get(country, country)
                            if iso_code and iso_code not in countries_list[idx]:
                                countries_list[idx].append(iso_code)
                        
                        if subregion and subregion not in subregions_list[idx]:
                            subregions_list[idx].append(subregion)
                            
                        if continent and continent not in continents_list[idx]:
                            continents_list[idx].append(continent)
        else:
            # Process sequentially
            logger.info("Processing LLM rows sequentially")
            
            for idx in tqdm(range(len(df)), desc="LLM processing"):
                _, llm_results = process_row_with_llm(idx)
                
                # Add any new results from LLM, avoiding duplicates
                for continent, subregion, country in llm_results:
                    if country:
                        # Convert to ISO code if possible
                        iso_code = iso2_country_code.get(country, country)
                        if iso_code and iso_code not in countries_list[idx]:
                            countries_list[idx].append(iso_code)
                    
                    if subregion and subregion not in subregions_list[idx]:
                        subregions_list[idx].append(subregion)
                        
                    if continent and continent not in continents_list[idx]:
                        continents_list[idx].append(continent)
    
    # STEP 3: Convert lists to comma-separated strings with better duplicate handling
    # First deduplicate the lists themselves
    for idx in range(len(df)):
        # Remove duplicates while preserving order
        if countries_list[idx]:
            countries_list[idx] = list(dict.fromkeys(countries_list[idx]))
        if subregions_list[idx]:
            subregions_list[idx] = list(dict.fromkeys(subregions_list[idx]))
        if continents_list[idx]:
            continents_list[idx] = list(dict.fromkeys(continents_list[idx]))
    
    # Now convert to strings
    countries_str = [', '.join(sorted(set(countries))) if countries else '' 
                     for countries in countries_list]
    subregions_str = [', '.join(sorted(set(subregions))) if subregions else '' 
                      for subregions in subregions_list]
    continents_str = [', '.join(sorted(set(continents))) if continents else '' 
                      for continents in continents_list]
    
    # Insert columns at the specified positions
    df.insert(loc=insert_pos, column='country', value=countries_str)
    df.insert(loc=insert_pos+1, column='subregion', value=subregions_str)
    df.insert(loc=insert_pos+2, column='continent', value=continents_str)
    
    return df


# -------------------- Country Standardization Function --------------------

import pandas as pd
import pycountry
import re
import logging

# Get the logger instance defined at the top level of the script
# This assumes logger = logging.getLogger(__name__) exists earlier
logger = logging.getLogger(__name__)

# Updated standardize_country_columns function code (extracted from country_cleanup.py)
def standardize_country_columns(df):
    """
    Standardize country names in DataFrame columns to ISO3 codes.
    Consolidates duplicate country columns and handles special cases.

    Args:
        df: pandas DataFrame with country names in column names

    Returns:
        DataFrame with standardized country column names
    """
    # Using the logger defined in the main script scope
    logger.info("Starting country column standardization...")

    # Step 1: Create comprehensive ISO3 mapping from pycountry
    country_name_to_iso3 = {}
    for country in pycountry.countries:
        country_name_to_iso3[country.name.lower()] = country.alpha_3
        country_name_to_iso3[country.alpha_2.lower()] = country.alpha_3
        if hasattr(country, "official_name"):
            country_name_to_iso3[country.official_name.lower()] = country.alpha_3
        if hasattr(country, "common_name"):
            country_name_to_iso3[country.common_name.lower()] = country.alpha_3

    # Step 2: Add manual overrides for historical/ambiguous names
    manual_iso3_map = {
        'BURMA': 'MMR', 'BYELORUSSIAN SSR': 'BLR', 'CAPE VERDE': 'CPV',
        'CENTRAL AFRICAN EMPIRE': 'CAF', 'CEYLON': 'LKA', "COTE D'IVOIRE": 'CIV',
        'DAHOMEY': 'BEN', 'DEMOCRATIC KAMPUCHEA': 'KHM', 'FEDERATION OF MALAYA': 'MYS',
        'GERMAN DEMOCRATIC REPUBLIC': 'DEU', 'FEDERAL REPUBLIC OF GERMANY': 'DEU', 'GERMANY': 'DEU',
        'IRAN (ISLAMIC REPUBLIC OF)': 'IRN', 'IVORY COAST': 'CIV', 'KHMER REPUBLIC': 'KHM',
        'MALDIVE ISLANDS': 'MDV', 'MICRONESIA (FEDERATED STATES OF)': 'FSM', 'PHILIPPINE REPUBLIC': 'PHL',
        'REPUBLIC OF KOREA': 'KOR', 'SIAM': 'THA', 'SURINAM': 'SUR', 'SWAZILAND': 'SWZ',
        'SYRIAN ARAB REPUBLIC': 'SYR', 'TANGANYIKA': 'TZA', 'THE FORMER YUGOSLAV REPUBLIC OF MACEDONIA': 'MKD',
        'TÜRKIYE': 'TUR', 'TÜRKÝYE': 'TUR', 'TÜRKİYE': 'TUR', 'TURKEY': 'TUR', 'UKRAINIAN SSR': 'UKR',
        'UNITED ARAB REPUBLIC': 'EGY', 'UPPER VOLTA': 'BFA', 'USSR': 'RUS', 'YUGOSLAVIA': 'SRB',
        'ZAIRE': 'COD', 'ZANZIBAR': 'TZA', 'CZECHOSVK': 'CZE', 'DEMOCRATIC YEMEN': 'YEM',
        'SOUTHERN YEMEN': 'YEM', 'UNITED CAMEROON': 'CMR', 'UNION OF SOUTH AFRICA': 'ZAF',
        'SERBIA AND MONTENEGRO': 'SRB', 'CONGO (BRAZZAVILLE)': 'COG', 'CONGO (DEMOCRATIC REPUBLIC OF)': 'COD',
        'CONGO (LEOPOLDVILLE)': 'COD', 'DEMOCRATIC CONGO': 'COD', 'VENEZUELA (BOLIVARIAN REPUBLIC OF)': 'VEN',
        'BOLIVIA (PLURINATIONAL STATE OF)': 'BOL', 'NETHERLANDS (KINGDOM OF THE)': 'NLD',
        'LIBYAN ARAB JAMAHIRIYA': 'LBY', 'LIBYAN ARAB REPUBLIC': 'LBY', 'DEMOCRATIC REPUBLIC OF THE CONGO': 'COD',
        # Lowercase variations often seen
        'bol (plurinational state of)': 'BOL', 'cog (brazzaville)': 'COG', 'cog (democratic republic of)': 'COD',
        'cog (leopoldville)': 'COD', 'democratic cog': 'COD', 'democratic yemen': 'YEM',
        'deu, federal republic of': 'DEU', '"deu, federal republic of"': 'DEU', 'iran (islamic republic of)': 'IRN',
        'libyan arab jamahiriya': 'LBY', 'libyan arab republic': 'LBY', 'nld (kingdom of the)': 'NLD',
        'serbia and montenegro': 'SRB', 'southern yemen': 'YEM', 'union of south africa': 'ZAF',
        'united cameroon': 'CMR', 'venezuela (bolivarian republic of)': 'VEN', 'micronesia (federated states of)': 'FSM',
        'democratic yem': 'YEM', 'lbyn arab jamahiriya': 'LBY', 'lbyn arab republic': 'LBY',
        'srb and mne': 'SRB', 'southern yem': 'YEM', 'union of zaf': 'ZAF', 'united cmr': 'CMR',
        'ven (bolivarian republic of)': 'VEN', 'irn (islamic republic of)': 'IRN', 'iran islamic republic of': 'IRN'
    }

    temp_manual_map = {k.lower(): v for k, v in manual_iso3_map.items()}
    country_name_to_iso3.update(temp_manual_map)

    # Step 3: Process each column name
    column_mapping = {}
    # Define fixed columns expected in the pipeline output
    initial_fixed_cols = set(['id', 'Council', 'Date', 'Title', 'Resolution', 'country',
                              'subregion', 'continent', 'tags', 'TOTAL VOTES', 'NO-VOTE COUNT',
                              'ABSTAIN COUNT', 'NO COUNT', 'YES COUNT', 'Link', 'token', 'Scrape_Year'])
    processed_fixed_cols = []
    unmapped_cols = []
    known_non_countries = {'YES', 'NO'} # Add others if they appear as 3-letter columns

    logger.info("Mapping original columns to ISO3 codes...")
    for col in df.columns:
        if col in initial_fixed_cols:
            processed_fixed_cols.append(col)
            continue # It's a fixed column

        iso3_code = None
        # Handle potential non-string column names gracefully
        if not isinstance(col, str):
            logger.warning(f"Skipping non-string column name: {col}")
            unmapped_cols.append(str(col)) # Add original column name (as string) to unmapped
            continue

        # --- Revised Mapping Logic for Step 3 ---
        col_to_check = col # Start with the original column name
        base_col_name = col
        is_suffixed = False

        # Check if it might be a suffixed column (e.g., 'DEU.1')
        if '.' in col and col.rsplit('.', 1)[1].isdigit():
            base_col_name = col.rsplit('.', 1)[0]
            logger.info(f"Column '{col}' looks suffixed. Checking base '{base_col_name}'.")
            col_to_check = base_col_name # Use the base name for mapping lookup
            is_suffixed = True

        col_clean = col_to_check.strip().lower()

        # Priority 1: Check cleaned base name against combined map
        if col_clean in country_name_to_iso3:
            iso3_code = country_name_to_iso3[col_clean]
            if is_suffixed:
                 logger.info(f"Mapped suffixed column '{col}' to {iso3_code} via base name '{base_col_name}'.")

        # Priority 2: Check original base column name if 3-letters uppercase
        elif len(col_to_check) == 3 and col_to_check.isupper():
            # Double check it's not a known non-country code mistaken as ISO3
            known_non_countries = {'YES', 'NO'} # Add others if needed
            if col_to_check not in known_non_countries:
                 iso3_code = col_to_check
                 if is_suffixed:
                      logger.info(f"Mapped suffixed column '{col}' to {iso3_code} via base ISO3 '{base_col_name}'.")
            elif not is_suffixed: # Only log skipping if it wasn't a suffix check
                 logger.info(f"Column '{col_to_check}' looks like ISO3 but is in known_non_countries set. Skipping.")

        # --- End of Revised Mapping Logic ---

        # Assign to mapping or track as unmapped
        if iso3_code:
            if iso3_code not in column_mapping:
                column_mapping[iso3_code] = []
            # IMPORTANT: Store the *original* column name (e.g., 'DEU.1')
            # associated with the identified iso3_code
            column_mapping[iso3_code].append(col)
        elif not is_suffixed: # Only track as unmapped if it wasn't a suffix we failed to map
            # Log only if it wasn't already identified as a fixed column
            logger.warning(f"MAP FAIL: Col='{col}', Cleaned='{col_clean}'. Tracking as unmapped.")
            unmapped_cols.append(col)
        elif is_suffixed and iso3_code is None:
             # Log if it looked like a suffix but the base couldn't be mapped
             logger.warning(f"MAP FAIL (Suffix): Col='{col}', Base='{base_col_name}' could not be mapped. Tracking as unmapped.")
             unmapped_cols.append(col)


    # Step 4: Consolidate duplicate columns
    logger.info("Consolidating columns...")
    # Keep only fixed columns present in the original DataFrame
    processed_fixed_cols = [col for col in processed_fixed_cols if col in df.columns]
    new_df = df[processed_fixed_cols].copy()
    country_data_collected = {}

    # --- Revised Consolidation Logic ---
    # Get all unique target ISO3 codes identified
    all_target_iso3_codes = set(column_mapping.keys())

    for iso3_code in all_target_iso3_codes:
        # Start with columns that were explicitly mapped (including suffixed ones mapped in Step 3)
        cols_to_combine = column_mapping.get(iso3_code, [])

        # Explicitly check if a column with the exact ISO3 name exists
        # and add it to the list if it's not already there from the mapping
        if iso3_code in df.columns and iso3_code not in cols_to_combine:
            # Insert at the beginning so the 'correct' name is prioritized by combine_first
            cols_to_combine.insert(0, iso3_code)

        # Ensure we only work with columns that actually exist in the original df
        valid_original_cols = [col for col in cols_to_combine if col in df.columns]

        if not valid_original_cols:
            logger.warning(f"No valid columns found to combine for ISO3 code {iso3_code} from list: {cols_to_combine}")
            continue

        logger.info(f"Combining columns for {iso3_code}: {valid_original_cols}")

        # Combine values using combine_first, starting with the first column in the list
        combined_series = df[valid_original_cols[0]].copy()
        if len(valid_original_cols) > 1:
            for col in valid_original_cols[1:]:
                # combine_first fills NaN in `combined_series` with values from `df[col]`
                combined_series = combined_series.combine_first(df[col])

        country_data_collected[iso3_code] = combined_series
    # --- End of Revised Logic ---


    if country_data_collected:
        country_df = pd.DataFrame(country_data_collected)
        new_df = pd.concat([new_df, country_df], axis=1)
        logger.info(f"Consolidated data for {len(country_data_collected)} ISO3 codes.")
    else:
        logger.info("No country columns were successfully mapped and consolidated.")

    unmapped_cols_present = [col for col in unmapped_cols if col in df.columns]
    if unmapped_cols_present:
         logger.warning(f"Adding {len(unmapped_cols_present)} unmapped columns to the end: {unmapped_cols_present}")
         # Filter unmapped_cols_present to ensure no duplicates with already added fixed/country cols
         unmapped_to_add = [col for col in unmapped_cols_present if col not in new_df.columns]
         if unmapped_to_add:
              new_df = pd.concat([new_df, df[unmapped_to_add]], axis=1)
         else:
              logger.info("Unmapped columns were already present (likely fixed columns).")


    # Final check to remove remaining suffixed columns (like 'DEU.1')
    # This step might now be redundant if Step 3 correctly maps suffixes, but keep for safety.
    logger.info("Checking for suffixed columns to drop...")
    current_columns = set(new_df.columns)
    cols_to_drop = []
    iso3_codes_present = set(country_data_collected.keys()) # Use keys from collected data

    for col in current_columns:
        if isinstance(col, str) and '.' in col:
            # Check if this suffixed column name itself exists in the mapping's values
            # (meaning it was successfully mapped in Step 3 but might still be present if base also exists)
            is_mapped_suffix = any(col in mapped_list for mapped_list in column_mapping.values())

            if is_mapped_suffix:
                 base_col_name = col.rsplit('.', 1)[0]
                 # Find the ISO code this suffix was mapped to
                 mapped_iso3_for_suffix = None
                 for iso, cols_list in column_mapping.items():
                      if col in cols_list:
                           mapped_iso3_for_suffix = iso
                           break
                 # If the primary ISO code column (e.g., 'DEU') also exists, drop the suffixed one ('DEU.1')
                 if mapped_iso3_for_suffix and mapped_iso3_for_suffix in current_columns and col != mapped_iso3_for_suffix:
                      cols_to_drop.append(col)
                      logger.info(f"Identified mapped suffixed column '{col}' whose base ISO '{mapped_iso3_for_suffix}' also exists. Marking for drop.")
            else:
                 # If the suffix itself wasn't mapped, check its base like before
                 base_col_name = col.rsplit('.', 1)[0]
                 if base_col_name in current_columns:
                      cols_to_drop.append(col)
                      logger.info(f"Identified unmapped suffixed column '{col}' whose base '{base_col_name}' exists. Marking for drop.")


    if cols_to_drop:
        cols_to_drop = list(set(cols_to_drop))
        logger.warning(f"Dropping identified suffixed/duplicate columns: {cols_to_drop}")
        cols_to_drop_existing = [c for c in cols_to_drop if c in new_df.columns]
        if cols_to_drop_existing:
             new_df = new_df.drop(columns=cols_to_drop_existing)

    # Step 5: Reorder columns
    logger.info("Reordering columns...")
    fixed_cols_present = [col for col in processed_fixed_cols if col in new_df.columns]
    country_cols_present = [
        col for col in new_df.columns
        if isinstance(col, str) and len(col) == 3 and col.isupper() and col not in initial_fixed_cols
    ]
    country_cols_present.sort()
    # Make sure 'unmapped_cols_present' contains only columns actually in the final df
    unmapped_cols_in_final = [col for col in unmapped_cols_present if col in new_df.columns]
    other_cols = [col for col in new_df.columns if col not in fixed_cols_present and col not in country_cols_present and col not in unmapped_cols_in_final]

    # Add the unmapped columns back at the end
    final_column_order = fixed_cols_present + country_cols_present + other_cols + unmapped_cols_in_final

    # Ensure final_column_order only contains columns that *actually* exist in new_df before reindexing
    final_column_order_existing = [col for col in final_column_order if col in new_df.columns]

    # Explicitly drop original mapped columns if the target ISO3 exists
    logger.info("Checking for original mapped columns to drop...")
    cols_to_drop_mapped = []
    current_columns_set = set(new_df.columns)
    for iso3_code, original_cols in column_mapping.items():
        # Check if the target ISO3 code exists in the DataFrame
        if iso3_code in current_columns_set:
            # Check each original column name that was mapped to this ISO3
            for original_col in original_cols:
                # If the original column name still exists and is different from the ISO3 code, mark it for dropping
                if original_col in current_columns_set and original_col != iso3_code:
                    cols_to_drop_mapped.append(original_col)
                    logger.info(f"Identified original column '{original_col}' mapped to '{iso3_code}'. Marking '{original_col}' for drop as '{iso3_code}' exists.")

    if cols_to_drop_mapped:
        cols_to_drop_mapped = list(set(cols_to_drop_mapped)) # Deduplicate
        cols_to_drop_existing = [c for c in cols_to_drop_mapped if c in new_df.columns]
        if cols_to_drop_existing:
            logger.warning(f"Dropping original columns that were mapped to existing ISO3 codes: {cols_to_drop_existing}")
            new_df = new_df.drop(columns=cols_to_drop_existing)
            # Recalculate final_column_order_existing after dropping columns
            final_column_order_existing = [col for col in final_column_order if col in new_df.columns]


    if list(new_df.columns) != final_column_order_existing:
        logger.info("Reordering columns to place sorted ISO3 codes after fixed columns.")
        new_df = new_df[final_column_order_existing]

    logger.info("Column standardization complete.")
    return new_df


# -------------------- Traditional Tagging Pipeline Functions --------------------

# Pydantic models for API responses
class MainTagClassification(BaseModel):
    """Pydantic model for stage 1 classification (main tags)"""
    main_tags: List[str] = Field(default_factory=list, description="List of relevant main category tags")

class SubTag1Classification(BaseModel):
    """Pydantic model for stage 2 classification (subtags)"""
    subtag1s: List[str] = Field(default_factory=list, description="List of relevant subcategories for the main tag")

class SubTag2Classification(BaseModel):
    """Pydantic model for stage 3 classification (specific items)"""
    subtag2s: List[str] = Field(default_factory=list, description="List of relevant specific items for the subcategory")

def flatten(lst):
    """Recursively flatten nested lists."""
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

def call_api_staged(title: str, stage: int, previous_tags: Optional[Dict] = None, 
                    model: str = DEFAULT_MODEL) -> Any:
    """
    Analyzes a UN resolution text in stages.
    
    Args:
        title: Title of the resolution to analyze
        stage: 1 for main tag, 2 for subtag1, 3 for subtag2
        previous_tags: Results from previous stages
        model: The OpenAI model to use
        
    Returns:
        Structured classification results
    """
    if stage == 1:
        # Stage 1: identify main tag categories
        main_tag_options = list(un_classification.keys())
        system_prompt = f"""You are a UN document classification assistant. Your task is to analyze UN resolutions given their Title.
Classify the resolution according to the following valid main categories (select only values from the list):
        
{main_tag_options}

Rules:
1. Identify ALL relevant main categories from the list.
2. Return only valid category names as a list.
3. If none of the categories apply, return an empty list.
"""
        api_call = lambda client: client.beta.chat.completions.parse(
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Resolution text: {title}"}
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            response_format=MainTagClassification,
        )
        try:
            response = execute_api_call(api_call)
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Error during main tag API call: {e}")
            return MainTagClassification(main_tags=[])
        
    elif stage == 2:
        # Stage 2: identify subtag1 based on main tags
        if not previous_tags or "main_tag" not in previous_tags:
            logger.error("Missing main_tag in previous_tags for stage 2 classification")
            return SubTag1Classification(subtag1s=[])
            
        main_tag = previous_tags["main_tag"]
        
        # Check if the main_tag exists in the classification dictionary
        if main_tag not in un_classification:
            logger.warning(f"Main tag '{main_tag}' not found in classification dictionary")
            return SubTag1Classification(subtag1s=[])
            
        subcategories = list(un_classification[main_tag].keys())
        
        system_prompt = f"""You are a UN document classification assistant. Your task is to analyze UN resolutions given their Title.
For a resolution categorized in the main category '{main_tag}', select the relevant subcategories from the following valid list:
        
{subcategories}

Rules:
1. Select only unique, valid subcategories from the list above.
2. If none of the listed subcategories apply, return an empty string.
3. Return only the valid subcategory names as a list.
"""
        api_call = lambda client: client.beta.chat.completions.parse(
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Resolution text: {title}"}
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            response_format=SubTag1Classification,
        )
        try:
            response = execute_api_call(api_call)
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Error during subtag1 API call for {main_tag}: {e}")
            return SubTag1Classification(subtag1s=[])
        
    elif stage == 3:
        # Stage 3: identify subtag2 based on main tag and subtag1
        if not previous_tags or "main_tag" not in previous_tags or "subtag1" not in previous_tags:
            logger.error("Missing required tags in previous_tags for stage 3 classification")
            return SubTag2Classification(subtag2s=[])
            
        main_tag = previous_tags["main_tag"]
        subtag1 = previous_tags["subtag1"]
        
        if main_tag not in un_classification or subtag1 not in un_classification[main_tag]:
            logger.error(f"Invalid tag combination: {main_tag} > {subtag1}")
            return SubTag2Classification(subtag2s=[])
            
        specific_items = un_classification[main_tag][subtag1]
        
        system_prompt = f"""You are a UN document classification assistant. Your task is to analyze UN resolutions given their Title.
For a resolution categorized as '{main_tag}' > '{subtag1}', choose the most relevant specific items from the following valid options:
        
{specific_items}

Rules:
1. Select only valid items from the above list.
2. If none of the specific items are applicable, return an empty list.
3. Return only valid items as a list.
"""
        api_call = lambda client: client.beta.chat.completions.parse(
            model=model,
            temperature=DEFAULT_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Resolution text: {title}"}
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            response_format=SubTag2Classification,
        )
        try:
            response = execute_api_call(api_call)
            return response.choices[0].message.parsed
        except Exception as e:
            logger.error(f"Error during subtag2 API call for {main_tag} > {subtag1}: {e}")
            return SubTag2Classification(subtag2s=[])
    
    else:
        logger.error(f"Invalid stage: {stage}")
        return None


def get_tags_sequential(title: str, model: str = DEFAULT_MODEL) -> List[List]:
    """
    Gets classification tags for a UN resolution using sequential processing within a row.
    This function returns one record per unique (main_tag, subtag1) pair.
    
    Args:
        title: The resolution title to classify.
        model: OpenAI model to use.
        
    Returns:
        List of lists containing:
          - [main_tag, subtag1, first_subtag2]
    """
    start_time = time.time()
    final_results = []
    
    # Stage 1: Get main tags.
    main_tags_result = call_api_staged(title, stage=1, model=model)
    if not main_tags_result.main_tags:
        logger.warning(f"No main tags found for: {title[:50]}...")
        return []
    
    # Process each main tag sequentially.
    for main_tag in main_tags_result.main_tags:
        # Skip the GEOGRAPHICAL DESCRIPTORS since we handle this with the geo_tagger
        if main_tag == "GEOGRAPHICAL DESCRIPTORS":
            continue
            
        # Stage 2: Get subtag1 results.
        subtag1_result = call_api_staged(
            title, 
            stage=2, 
            previous_tags={"main_tag": main_tag},
            model=model
        )
        
        if not subtag1_result.subtag1s:
            logger.debug(f"No subtag1s found for main tag: {main_tag}")
            continue
        
        # Deduplicate subtag1 values (preserving order)
        unique_subtag1 = []
        for subtag1 in subtag1_result.subtag1s:
            if subtag1 not in unique_subtag1:
                unique_subtag1.append(subtag1)
        
        # Process each unique subtag1 sequentially.
        for subtag1 in unique_subtag1:
            # Stage 3: Get subtag2 results.
            subtag2_result = call_api_staged(
                title, 
                stage=3, 
                previous_tags={"main_tag": main_tag, "subtag1": subtag1},
                model=model
            )
            
            # For non-geographic descriptors, only take the first subtag2 value.
            first_subtag2 = subtag2_result.subtag2s[0] if subtag2_result.subtag2s else None
            final_results.append([main_tag, subtag1, first_subtag2])
    
    elapsed_time = time.time() - start_time
    logger.debug(f"Classification completed in {elapsed_time:.2f}s")
    
    return final_results

def tag_resolution(title: str, model: str = DEFAULT_MODEL):
    """
    Get classification tags for a resolution title.
    Returns the string representation of the list-of-lists.
    """
    tags = get_tags_sequential(title, model=model)
    return str(tags)

def process_tags(tag_str):
    """
    Convert a string representation of a list-of-lists (from the tagging API)
    into a comma-separated string for the 'tags' column.
    
    The 'geographic' data is now handled by the geo_tagger.
    """
    try:
        tag_lists = ast.literal_eval(tag_str)
    except Exception:
        return ""
        
    non_geo = []
    for sub in tag_lists:
        if isinstance(sub, list) and sub:
            non_geo.extend(flatten(sub))
        else:
            non_geo.append(sub)
            
    non_geo_str = ', '.join([str(x) for x in non_geo if x is not None])
    return non_geo_str


# -------------------- Combined Tagging Function --------------------

def tag_new_rows(new_df, geo_hierarchy, iso2_country_code, model=DEFAULT_MODEL, max_workers=DEFAULT_MAX_WORKERS):
    """
    Process new rows using both traditional tagging and geo-tagging.
    """
    logger.info(f"Tagging {len(new_df)} new rows...")
    
    # Step 1: Apply traditional tagging function to each Title
    new_df['tags_raw'] = new_df['Title'].apply(lambda t: tag_resolution(t, model=model))
    
    # Step 2: Process tags to extract non-geographic tags
    new_df['tags'] = new_df['tags_raw'].apply(process_tags)
    
    # Remove temporary column
    new_df.drop('tags_raw', axis=1, inplace=True)
    
    # Step 3: Apply geo-tagging to add country, subregion, continent columns
    new_df = combined_geo_tagger(
        df=new_df,
        geo_hierarchy=geo_hierarchy,
        iso2_country_code=iso2_country_code,
        model=model,
        max_workers=max_workers,
        use_llm=True
    )
    
    logger.info("Tagging complete.")
    return new_df


# -------------------- Scraper Pipeline Functions --------------------

def get_driver():
    """Initializes and returns a Selenium WebDriver instance for a headless environment."""
    options = Options()
    options.add_argument(f"user-agent={random.choice(USER_AGENTS)}")
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    
    # Set up the service using WebDriver Manager
    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    except Exception as e:
        logger.error(f"Failed to initialize WebDriver: {e}")
        raise

def normalize_link(href):
    """Normalize a UN record URL from the given href."""
    if not href:
        return None
    if '/record/' in href:
        try:
            record_part = href.split('/record/')[1]
            record_id = record_part.split('?')[0].split('/')[0].strip()
            if record_id.isdigit():
                return f"https://digitallibrary.un.org/record/{record_id}"
        except (IndexError, ValueError):
            pass
    base_url = href.split('?')[0]
    if '?' in href:
        params = href.split('?')[1].split('&')
        ln_param = [p for p in params if p.startswith('ln=')]
        if ln_param:
            base_url = f"{base_url}?{ln_param[0]}"
    return base_url

def get_links_from_csv_regex(csv_file):
    """
    Extract UN record links from the master CSV using a regex.
    Returns a list of unique normalized links.
    """
    links = set()
    pattern = re.compile(r'https://digitallibrary\.un\.org/record/\d+')
    if not os.path.exists(csv_file):
        return []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f:
                found = pattern.findall(line)
                for link in found:
                    norm = normalize_link(link)
                    if norm:
                        links.add(norm)
        logger.info(f"Extracted {len(links)} unique links from CSV for deduplication.")
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
    return list(links)

def extract_vote_data_from_html(html_content):
    """Extract vote data and metadata from the page HTML."""
    soup = BeautifulSoup(html_content, "html.parser")
    data = {}
    try:
        script_tag = soup.find('script', {'type': 'application/ld+json', 'id': 'detailed-schema-org'})
        if script_tag and script_tag.string:
            json_data = json.loads(script_tag.string)
            data['Title'] = json_data.get('name', '')
            data['Date'] = json_data.get('datePublished', '')
    except Exception:
        pass
    for row in soup.find_all('div', class_='metadata-row'):
        try:
            title_elem = row.find('span', class_='title')
            value_elem = row.find('span', class_='value')
            if title_elem and value_elem:
                title_text = title_elem.text.strip()
                if title_text == 'Vote':
                    value = value_elem.get_text('\n').strip()
                    vote_data = {}
                    for line in value.split('\n'):
                        line = line.strip()
                        if line:
                            parts = re.match(r'^\s*([YNA])\s+(.+)', line)
                            if parts:
                                vote_data[parts.group(2).strip().upper()] = parts.group(1).upper()
                    data['Vote Data'] = vote_data
                else:
                    data[title_text] = value_elem.text.strip()
        except Exception:
            continue
    return data

def determine_council(title):
    """Determine the council type based on the resolution title."""
    if not title:
        return "Unknown"
    lower_title = title.lower()
    if "security council" in lower_title or "S/RES/" in title:
        return "Security Council"
    if "general assembly" in lower_title or "A/RES/" in title:
        return "General Assembly"
    return "Unknown"

def process_resolution(link, driver, year):
    """
    Process a single resolution page and return a dictionary of row data.
    """
    try:
        record_id = link.split('/record/')[1].split('?')[0] if '/record/' in link else link.split('/')[-1]
        logger.info(f"Processing record: {record_id}")
        driver.get(link)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'metadata-row')]"))
        )
        row_data = {"Link": link, "token": record_id, "Scrape_Year": year}
        html_content = driver.page_source
        extracted = extract_vote_data_from_html(html_content)
        if extracted:
            if extracted.get('Title'):
                row_data['Title'] = extracted['Title']
                row_data['Council'] = determine_council(extracted['Title'])
            if extracted.get('Resolution'):
                row_data['Resolution'] = extracted['Resolution']
            if extracted.get('Vote date'):
                row_data['Date'] = extracted['Vote date']
            if extracted.get('Vote summary'):
                summary = extracted['Vote summary']
                if m := re.search(r'Yes:\s*(\d+)', summary):
                    row_data['YES COUNT'] = m.group(1)
                if m := re.search(r'No:\s*(\d+)', summary):
                    row_data['NO COUNT'] = m.group(1)
                if m := re.search(r'Abstentions:\s*(\d+)', summary):
                    row_data['ABSTAIN COUNT'] = m.group(1)
                if m := re.search(r'Non-Voting:\s*(\d+)', summary):
                    row_data['NO-VOTE COUNT'] = m.group(1)
                if m := re.search(r'Total voting membership:\s*(\d+)', summary):
                    row_data['TOTAL VOTES'] = m.group(1)
            if 'Vote Data' in extracted:
                for country, vote in extracted['Vote Data'].items():
                    if vote == 'Y':
                        row_data[country] = 'YES'
                    elif vote == 'N':
                        row_data[country] = 'NO'
                    elif vote == 'A':
                        row_data[country] = 'ABSTAIN'
        if row_data.get('Title') or row_data.get('Resolution'):
            return row_data
        return None
    except Exception as e:
        logger.error(f"Error processing link {link}: {e}")
        return None

def batch_scrape_resolutions(links, driver, year, batch_size=15):
    """
    Scrape resolution pages in batches.
    Returns (successful_rows, failed_links).
    """
    batch_rows = []
    failed_links = []
    total_links = len(links)
    for i in range(0, total_links, batch_size):
        batch = links[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(total_links + batch_size - 1)//batch_size} with {len(batch)} links.")
        for link in batch:
            row_data = process_resolution(link, driver, year)
            if row_data:
                batch_rows.append(row_data)
            else:
                logger.warning(f"No data for link: {link}. Marking as failed.")
                failed_links.append(link)
            time.sleep(0.2)
        time.sleep(0.5)
    return batch_rows, failed_links

def parallel_scrape_resolutions(links, year, num_workers=2, batch_size=15):
    """
    Scrape resolution pages in parallel using multiple browser instances.
    Returns (all_rows, all_failed_links).
    """
    if not links:
        return [], []
    all_rows = []
    all_failed_links = []
    chunks = [links[i::num_workers] for i in range(num_workers)]
    def worker_task(worker_id, worker_links):
        worker_driver = get_driver()
        try:
            rows, failed = batch_scrape_resolutions(worker_links, worker_driver, year, batch_size)
            logger.info(f"Worker {worker_id} processed {len(worker_links)} links with {len(rows)} rows and {len(failed)} failures.")
            return rows, failed
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            return [], worker_links
        finally:
            worker_driver.quit()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task, i, chunk) for i, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            rows, failed = future.result()
            all_rows.extend(rows)
            all_failed_links.extend(failed)
    logger.info(f"Parallel scraping complete: {len(all_rows)} rows, {len(all_failed_links)} failed links.")
    return all_rows, all_failed_links

def check_for_next_button(driver):
    """Locate the 'next' button for pagination."""
    try:
        next_button = driver.find_element(By.XPATH, "//a[img[@alt='next']]")
        return next_button
    except NoSuchElementException:
        return None

def collect_links_for_year(driver, year, existing_links):
    """
    Paginate the search results for a given year and collect new links.
    Once a duplicate link is encountered on a page, finish collecting any new links
    from that page and then stop. If all links on a page are duplicates, stop immediately.
    
    Returns a list of new links (i.e. not in existing_links).
    """
    all_links = set()
    page_count = 0

    while page_count < MAX_PAGES_PER_YEAR:
        page_count += 1
        logger.info(f"[Year {year}] Processing page {page_count}.")
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/record/')]"))
            )
        except TimeoutException:
            logger.warning(f"Timeout on page {page_count} for year {year}.")
            break

        elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/record/')]")
        new_links_on_page = False
        duplicate_found = False
        page_links = []
        
        # First pass: collect all valid links from the page
        for elem in elements:
            try:
                href = elem.get_attribute("href")
                norm_link = normalize_link(href)
                if norm_link:
                    page_links.append(norm_link)
            except StaleElementReferenceException:
                continue
        
        # Second pass: process all links from the page
        for link in page_links:
            if link in existing_links:
                duplicate_found = True
            else:
                all_links.add(link)
                new_links_on_page = True
        
        # If we found any duplicates on this page
        if duplicate_found:
            if len(all_links) > 0:
                logger.info(f"[Year {year}] Found {len(all_links)} unique new links before duplicate.")
                raise DuplicateLinkFound(f"Duplicate link encountered in year {year}", list(all_links))
            else:
                logger.info(f"[Year {year}] No new links found before duplicate.")
                return []
        
        # If no new links were found on this page, stop processing
        if not new_links_on_page:
            logger.info(f"[Year {year}] No new links on page {page_count}; stopping collection.")
            break

        # Only continue to next page if we found new links and no duplicates
        next_button = check_for_next_button(driver)
        if next_button:
            try:
                driver.execute_script("arguments[0].scrollIntoView();", next_button)
                time.sleep(0.2)
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(1)
            except Exception as e:
                logger.error(f"[Year {year}] Error clicking next button: {e}")
                break
        else:
            logger.info(f"[Year {year}] No next button found; reached last page.")
            break

    logger.info(f"[Year {year}] Collected {len(all_links)} new links.")
    return list(all_links)

def get_available_years(driver):
    """Extract available years and their counts from the date facet."""
    date_facets = []
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//ul[contains(@class, 'option-fct')]"))
        )
        date_headers = driver.find_elements(By.XPATH, "//h2[text()='Date']")
        for header in date_headers:
            try:
                facet_section = header.find_element(By.XPATH, "./following-sibling::ul[contains(@class, 'option-fct')]")
                if "expanded" not in facet_section.get_attribute("class"):
                    try:
                        show_more = header.find_element(By.XPATH, "./following-sibling::span[contains(@class, 'showmore')]")
                        driver.execute_script("arguments[0].click();", show_more)
                        time.sleep(0.5)
                    except (NoSuchElementException, ElementNotInteractableException):
                        pass
                year_inputs = facet_section.find_elements(By.XPATH, ".//input[@type='checkbox']")
                for inp in year_inputs:
                    year_value = inp.get_attribute("value")
                    try:
                        label = driver.find_element(By.XPATH, f"//label[@for='{inp.get_attribute('id')}']")
                        year_text = label.text.strip()
                        match = re.match(r'(\d{4})\s*\((\d+)\)', year_text)
                        if match:
                            year, count = match.group(1), int(match.group(2))
                            date_facets.append({
                                'year': year,
                                'count': count,
                                'input_id': inp.get_attribute('id'),
                                'input_value': year_value
                            })
                    except NoSuchElementException:
                        continue
            except Exception as e:
                logger.error(f"Error processing date header: {e}")
                continue
        return sorted(date_facets, key=lambda x: x['year'], reverse=True)
    except Exception as e:
        logger.error(f"Error getting available years: {e}")
        return []

def select_year_facet(driver, year_data, max_retries=10):
    """
    Select a specific year by clicking its checkbox.
    If "no such element" errors occur five times, refresh the browser session (switching user agent).
    Returns a tuple: (True/False, driver)
    """
    no_element_error_count = 0

    for retry in range(max_retries):
        try:
            logger.info(f"Selecting year: {year_data['year']} (Attempt {retry+1}/{max_retries})")
            checkbox = driver.find_element(By.ID, year_data['input_id'])
            driver.execute_script("arguments[0].scrollIntoView();", checkbox)
            time.sleep(0.2)
            driver.execute_script("arguments[0].click();", checkbox)
            time.sleep(1)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/record/')]"))
            )
            records = driver.find_elements(By.XPATH, "//a[contains(@href, '/record/')]")
            if records and len(records) > 0:
                logger.info(f"Selected year {year_data['year']} with {len(records)} visible records")
                return True, driver
            else:
                logger.warning(f"Year {year_data['year']} selected but no records visible")
                if retry < max_retries - 1:
                    clear_filters(driver)
        except Exception as e:
            error_message = str(e).lower()
            logger.error(f"Error selecting year {year_data['year']}: {e}")
            if "no such element" in error_message:
                no_element_error_count += 1
                logger.warning(f"'No such element' error count: {no_element_error_count}")
                if no_element_error_count >= 5:
                    logger.info("5 'no such element' errors encountered; switching user agent.")
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    driver = get_driver()
                    driver.get(BASE_SEARCH_URL)
                    time.sleep(2)
                    # Reset the error count after switching agent
                    no_element_error_count = 0
                    continue
            if retry < max_retries - 1:
                clear_filters(driver)
    try:
        logger.warning(f"Trying fallback for year {year_data['year']}...")
        driver.get(BASE_SEARCH_URL)
        time.sleep(1.5)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//h2[text()='Date']"))
        )
        checkbox = driver.find_element(By.ID, year_data['input_id'])
        driver.execute_script("arguments[0].click();", checkbox)
        time.sleep(1.5)
        records = driver.find_elements(By.XPATH, "//a[contains(@href, '/record/')]")
        if records and len(records) > 0:
            logger.info(f"Fallback: Selected year {year_data['year']} with {len(records)} visible records")
            return True, driver
    except Exception as e:
        logger.error(f"Fallback selection failed: {e}")
    logger.error(f"Failed to select year {year_data['year']} after multiple attempts")
    return False, driver

def clear_filters(driver):
    """Clear all filters to reset the search."""
    try:
        clear_buttons = driver.find_elements(
            By.XPATH, "//a[contains(text(), 'Clear') or contains(@class, 'clear') or contains(@onclick, 'clear')]"
        )
        if clear_buttons:
            for button in clear_buttons:
                try:
                    driver.execute_script("arguments[0].click();", button)
                    time.sleep(0.5)
                    return True
                except Exception:
                    continue
        driver.get(BASE_SEARCH_URL)
        time.sleep(1)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/record/')]"))
        )
        return True
    except Exception as e:
        logger.error(f"Error clearing filters: {e}")
        try:
            driver.get(BASE_SEARCH_URL)
            time.sleep(1)
            return True
        except:
            pass
        return False

def retry_failed_links(failed_links, year):
    """Retry processing failed links with a fresh browser session."""
    if not failed_links:
        return []

    logging.info(f"Retrying {len(failed_links)} failed links for year {year}...")

    retry_driver = get_driver()  # New session with rotated user-agent
    retried_rows = []

    try:
        for link in failed_links:
            try:
                row_data = process_resolution(link, retry_driver, year)
                if row_data:
                    retried_rows.append(row_data)
                time.sleep(0.2)
            except Exception as e:
                logging.error(f"Retry failed for link {link}: {e}")
    finally:
        retry_driver.quit()

    logging.info(f"Retried {len(failed_links)} links, successfully recovered {len(retried_rows)} records.")
    return retried_rows

def main():
    """Main function to orchestrate the scraping and tagging pipeline."""
    # Initialize Supabase client and get existing links
    supabase_client = create_supabase_client()
    existing_links = get_existing_links_from_supabase(supabase_client)

    # Initialize Selenium WebDriver
    driver = get_driver()
    
    # Get available years for scraping
    years_data = get_available_years(driver)
    
    # Sort years from oldest to newest to process chronologically
    sorted_years = sorted(years_data.keys(), key=int)
    
    all_new_data = []

    for year in sorted_years:
        logger.info(f"--- Processing year: {year} ---")
        
        # Collect all resolution links for the year, filtering against existing ones
        new_links = collect_links_for_year(driver, years_data[year], existing_links)
        
        if not new_links:
            logger.info(f"No new resolutions found for year {year}.")
            continue
            
        logger.info(f"Found {len(new_links)} new resolutions to scrape for year {year}.")

        # Scrape and process new resolutions in parallel
        scraped_data = parallel_scrape_resolutions(new_links, year, num_workers=MAX_WORKERS)
        
        if not scraped_data:
            logger.warning(f"No data was successfully scraped for year {year}.")
            continue

        # Convert scraped data to DataFrame
        new_df = pd.DataFrame(scraped_data)
        
        # --- Tagging and Classification for New Rows ---
        logger.info(f"Starting tagging process for {len(new_df)} new rows for year {year}...")
        
        # Perform geo-tagging and subject tagging
        tagged_df = tag_new_rows(new_df.copy(), geo_hierarchy, iso2_country_code, model=DEFAULT_MODEL, max_workers=MAX_WORKERS)
        
        # Standardize country columns to ISO codes
        final_df = standardize_country_columns(tagged_df)
        
        # Append to the list of all new data
        all_new_data.append(final_df)

    # Close the WebDriver
    driver.quit()

    if not all_new_data:
        logger.info("No new resolutions found across all years. Pipeline finished.")
        return

    # Combine all new data from all years into a single DataFrame
    combined_new_df = pd.concat(all_new_data, ignore_index=True)
    
    logger.info(f"Total new resolutions processed: {len(combined_new_df)}")

    # --- Upload new data to Supabase ---
    if not combined_new_df.empty:
        logger.info("Uploading new data to Supabase...")
        # Convert DataFrame to list of dictionaries for upload
        records_to_upload = combined_new_df.to_dict(orient='records')
        
        try:
            response = supabase_client.table('un_votes_raw').insert(records_to_upload).execute()
            if response.data:
                logger.info(f"Successfully uploaded {len(response.data)} new rows to Supabase.")
            else:
                logger.error(f"Failed to upload to Supabase. Response: {response}")
        except Exception as e:
            logger.error(f"An error occurred during Supabase upload: {e}")

    # --- Save a local copy of the new data ---
    today_str = datetime.now().strftime("%Y-%m-%d")
    new_data_filename = f"pipeline_output/NEW_DATA_{today_str}.csv"
    combined_new_df.to_csv(new_data_filename, index=False)
    logger.info(f"Saved a local copy of new data to {new_data_filename}")
    
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()

