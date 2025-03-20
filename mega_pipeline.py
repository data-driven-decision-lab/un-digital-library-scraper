#!/usr/bin/env python3
"""
Combined UN Resolution Scraper & Tagging Pipeline

This script:
  • Loads an existing master CSV (data/UN_VOTING_DATA_RAW.csv) containing all rows.
  • Runs the scraper to obtain new resolution rows (skipping rows already present).
  • Sends new rows for tagging via OpenAI using a three-stage classification.
  • Splits the tagging output into two new columns:
      - 'geographic' (from sublists starting with "GEOGRAPHICAL DESCRIPTORS")
      - 'tags' (for all other tags)
  • Merges new rows with existing data, sorts by Date (oldest row gets id 0),
    reassigns the id, and reorders columns so that 'geographic' and 'tags' appear immediately after 'Resolution' (with 'id' as the first column).
  • Writes out a final CSV file whose filename includes the current scrape date.
  
Requirements:
  - The scraper outputs all rows in a CSV.  
  - Only new rows (based on unique tokens) are processed for tagging.
  - The final CSV contains the complete set of rows.
  - Only one API key is used (set via the API_KEY environment variable).
"""

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
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from un_classification import un_classification
from typing import Optional, Dict, Any
from typing import List, Optional, Dict, Any, Tuple, Callable
from tqdm import tqdm
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from queue import Queue

import random
import logging
import os
import time
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple, Callable
import argparse





# Import classification schema
from un_classification import un_classification



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

# ---------------- OpenAI & Tagging Imports ----------------
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------- Configuration & Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),  # Console handler
        logging.FileHandler("un_scraper_tagger.log")  # File handler
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set your API key (only one needed)
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables.")

# ---------------- Global Settings ----------------
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1000
DEFAULT_MAX_WORKERS = 2  # For parallel scraping

# Create directories
os.makedirs("pipeline_output", exist_ok=True)

# Find the most recent CSV file in the pipeline_output folder
def get_latest_master_csv():
    # Look for all CSV files with the pattern UN_VOTING_DATA_RAW_WITH_TAGS_*.csv
    pattern = "pipeline_output/UN_VOTING_DATA_RAW_WITH_TAGS_*.csv"
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        # No existing files, create a default filename with today's date
        today = datetime.now().strftime("%Y-%m-%d")
        return f"pipeline_output/UN_VOTING_DATA_RAW_WITH_TAGS_{today}.csv"
    
    # Option 1: Sort by file modification time
    # latest_file = max(csv_files, key=os.path.getmtime)
    
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
    
    # Fallback: if date parsing fails, use modification time
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file

# Set the master CSV file
MASTER_CSV = get_latest_master_csv()
print(f"Using master CSV: {MASTER_CSV}")

FIXED_COLUMNS = [
    "Council", "Date", "Title", "Resolution", "TOTAL VOTES", "NO-VOTE COUNT",
    "ABSENT COUNT", "NO COUNT", "YES COUNT", "Link", "token", "Scrape_Year"
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


# -------------------- Tagging Pipeline Functions --------------------

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

def process_tags(tag_str):
    """
    Convert a string representation of a list-of-lists (from the tagging API)
    into two comma-separated strings: one for geographic info and one for other tags.
    """
    try:
        tag_lists = ast.literal_eval(tag_str)
    except Exception:
        return pd.Series(["", ""])
    geographic = []
    non_geo = []
    for sub in tag_lists:
        if isinstance(sub, list) and sub:
            if sub[0] == 'GEOGRAPHICAL DESCRIPTORS':
                geographic.extend(flatten(sub[1:]))
            else:
                non_geo.extend(flatten(sub))
        else:
            non_geo.append(sub)
    geo_str = ', '.join([str(x) for x in geographic if x is not None])
    non_geo_str = ', '.join([str(x) for x in non_geo if x is not None])
    return pd.Series([geo_str, non_geo_str])

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
    
    For "GEOGRAPHICAL DESCRIPTORS", all corresponding subtag2 values are aggregated into a list.
    For non geographic descriptors, only the first subtag2 value is returned.
    
    Args:
        title: The resolution title to classify.
        model: OpenAI model to use.
        
    Returns:
        List of lists containing:
          - For GEOGRAPHICAL DESCRIPTORS: [main_tag, subtag1, aggregated_subtag2_list]
          - For other main tags: [main_tag, subtag1, first_subtag2]
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
            
            if main_tag == "GEOGRAPHICAL DESCRIPTORS":
                # For geographic descriptors, aggregate all subtag2 values.
                subtag2_values = subtag2_result.subtag2s if subtag2_result.subtag2s else []
                final_results.append([main_tag, subtag1, subtag2_values])
            else:
                # For non geographic descriptors, only take the first subtag2 value.
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

def tag_new_rows(new_df, model="gpt-4o-mini"):
    """
    Process new rows using the tagging pipeline.
    For each row (using its Title), get classification tags,
    then split the result into 'geographic' and 'tags' columns.
    """
    logger.info(f"Tagging {len(new_df)} new rows...")
    
    # Apply tagging function to each Title
    new_df['tags'] = new_df['Title'].apply(lambda t: tag_resolution(t, model=model))
    
    # Process tags to split into geographic and non-geographic columns
    new_df[['geographic', 'tags']] = new_df['tags'].apply(process_tags)
    
    logger.info("Tagging complete.")
    return new_df

# -------------------- Scraper Pipeline Functions --------------------

def get_driver():
    """Initialize and return a Selenium Chrome driver with a rotated user-agent."""
    global user_agent_index
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    user_agent = USER_AGENTS[user_agent_index]
    user_agent_index = (user_agent_index + 1) % len(USER_AGENTS)
    options.add_argument(f"user-agent={user_agent}")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--js-flags=--expose-gc")
    options.add_argument("--aggressive-cache-discard")
    options.add_argument("--disable-site-isolation-trials")
    driver_path = ChromeDriverManager().install()
    try:
        os.chmod(driver_path, 0o755)
    except Exception as e:
        logger.warning(f"Could not set permissions for {driver_path}: {e}")
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(45)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    logger.info(f"Initialized browser with user-agent: {user_agent}")
    return driver

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
                            parts = re.match(r'^\s*([YNA])\s+(.+)$', line)
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
                    row_data['ABSENT COUNT'] = m.group(1)
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
    Once a duplicate link (one already in existing_links) is encountered,
    raise DuplicateLinkFound to immediately halt further processing.
    
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
        
        for elem in elements:
            try:
                href = elem.get_attribute("href")
                norm_link = normalize_link(href)
                if norm_link:
                    # Check if link is already in our existing links
                    if norm_link in existing_links:
                        logger.info(f"Duplicate link encountered ({norm_link}).")
                        duplicate_found = True
                    else:
                        all_links.add(norm_link)
                        new_links_on_page = True
            except StaleElementReferenceException:
                continue
                
        # If a duplicate was found on this page, raise the exception with all new links collected so far
        if duplicate_found:
            if len(all_links) > 0:
                logger.info(f"[Year {year}] Found {len(all_links)} unique new links before duplicate.")
                raise DuplicateLinkFound(f"Duplicate link encountered in year {year}", list(all_links))
            else:
                logger.info(f"[Year {year}] No new links found before duplicate.")
                return []
        
        # If no new links were found on this page (all were duplicates), we can stop
        if not new_links_on_page:
            logger.info(f"[Year {year}] No new links on page {page_count}; stopping collection.")
            break

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
    """
    Main function that integrates scraping and tagging:
      - Loads the master CSV (for reference) but does NOT modify it.
      - Scrapes new rows and merges them with the old data.
      - Tags only the new rows.
      - Saves the full dataset (new + old rows) into a new CSV file.
      - The master CSV remains unchanged.
    """
    # Create master CSV directory if needed.
    if os.path.dirname(MASTER_CSV):
        os.makedirs(os.path.dirname(MASTER_CSV), exist_ok=True)
    
    # Load existing master CSV (if it exists).
    if os.path.exists(MASTER_CSV):
        master_df = pd.read_csv(MASTER_CSV, dtype=str)
        logger.info(f"Loaded {len(master_df)} existing rows from master CSV.")
    else:
        master_df = pd.DataFrame()
        logger.info("No master CSV found; starting fresh.")
    
    # Get existing links from the master CSV for deduplication.
    csv_links = set(get_links_from_csv_regex(MASTER_CSV))
    logger.info(f"Loaded {len(csv_links)} unique links from CSV for deduplication.")
    
    # Initialize Selenium driver and load the base search page.
    driver = get_driver()
    driver.get(BASE_SEARCH_URL)
    time.sleep(2)
    
    years_data = get_available_years(driver)
    if not years_data:
        logger.error("No years found on the page. Check the website structure.")
        driver.quit()
        return
    logger.info(f"Found {len(years_data)} years to process")
    
    new_rows_all = []
    session_request_count = 0
    SESSION_RESET_THRESHOLD = 150

    try:
        for year_data in years_data:
            year = year_data['year']
            logger.info(f"\n{'='*60}\nProcessing year {year} ({year_data['count']} records)\n{'='*60}")
            
            # Refresh the session if threshold exceeded.
            if session_request_count > SESSION_RESET_THRESHOLD:
                driver.quit()
                driver = get_driver()
                driver.get(BASE_SEARCH_URL)
                time.sleep(2)
                session_request_count = 0
            
            # Use the facet selection function that handles errors and user-agent rotation.
            success, driver = select_year_facet(driver, year_data)
            if not success:
                logger.error(f"Failed to select facet for {year}; skipping to next year.")
                continue
            session_request_count += 1
            
            # Attempt to collect new links. If a duplicate is encountered, catch the exception.
            try:
                new_links = collect_links_for_year(driver, year, csv_links)
            except DuplicateLinkFound as e:
                # The new_links attribute should now contain all links found before the duplicate
                new_links = e.new_links
                logger.info(f"Duplicate link encountered; found {len(new_links)} new links before duplicate in year {year}")
                # We removed the stop_processing flag so that the loop will continue to the next year.
            
            if new_links:
                logger.info(f"Collected {len(new_links)} new links for year {year}")
                # Process links: use parallel scraping if many links; otherwise, batch process.
                BATCH_SIZE = 40
                if len(new_links) > 50 and MAX_WORKERS > 1:
                    logger.info(f"Using parallel processing with {MAX_WORKERS} workers")
                    batch_rows, failed_links = parallel_scrape_resolutions(new_links, year, MAX_WORKERS)
                    if batch_rows:
                        new_rows_all.extend(batch_rows)
                    if failed_links:
                        retry_rows = retry_failed_links(failed_links, year)
                        if retry_rows:
                            new_rows_all.extend(retry_rows)
                else:
                    for i in range(0, len(new_links), BATCH_SIZE):
                        session_request_count += 1
                        if session_request_count >= SESSION_RESET_THRESHOLD:
                            logger.info(f"Session reset threshold reached ({SESSION_RESET_THRESHOLD} requests)")
                            driver.quit()
                            driver = get_driver()
                            driver.get(BASE_SEARCH_URL)
                            time.sleep(2)
                            session_request_count = 0
                        batch_links = new_links[i:i+BATCH_SIZE]
                        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(new_links) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch_links)} links)")
                        batch_rows, failed_links = batch_scrape_resolutions(batch_links, driver, year)
                        if batch_rows:
                            new_rows_all.extend(batch_rows)
                        if failed_links:
                            retry_rows = retry_failed_links(failed_links, year)
                            if retry_rows:
                                new_rows_all.extend(retry_rows)
                
                # Update the in-memory set to avoid reprocessing these links.
                csv_links.update(new_links)
            else:
                logger.warning(f"No new links found for year {year}.")
            
            clear_filters(driver)
            time.sleep(1)

            # Removed the following break to ensure processing of all years:
            # if stop_processing:
            #     logger.info("Stopping further year processing due to duplicate link encountered.")
            #     break

    except Exception as general_e:
        logger.error(f"An error occurred during processing: {general_e}")
    finally:
        driver.quit()
    
    logger.info(f"Scraping complete. {len(new_rows_all)} new rows collected.")
    
    if not new_rows_all:
        logger.info("No new rows to process. Exiting.")
        return
    
    # Create DataFrame from new rows and tag only these new rows.
    new_df = pd.DataFrame(new_rows_all)
    tagged_new_df = tag_new_rows(new_df, model=DEFAULT_MODEL)
    
    # Merge new rows with master data to create a complete dataset
    if not master_df.empty:
        combined_df = pd.concat([master_df, tagged_new_df], ignore_index=True)
    else:
        combined_df = tagged_new_df
    
    # Standardize: sort by Date (oldest first), reassign id, and reorder columns.
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
    combined_df = combined_df.sort_values('Date', ascending=True).reset_index(drop=True)
    combined_df['id'] = combined_df.index
    cols = list(combined_df.columns)
    for col in ['geographic', 'tags']:
        if col in cols:
            cols.remove(col)
    if "Resolution" in cols:
        resolution_index = cols.index("Resolution")
    else:
        resolution_index = 3
    new_order = cols[:resolution_index+1] + ['geographic', 'tags'] + cols[resolution_index+1:]
    if 'id' in new_order:
        new_order.remove('id')
    new_order = ['id'] + new_order
    combined_df = combined_df[new_order]
    
    # Write final CSV with the current scrape date appended.
    scrape_date = datetime.now().strftime("%Y-%m-%d")
    output_csv = f"pipeline_output/UN_VOTING_DATA_RAW_with_tags_{scrape_date}.csv"
    combined_df.to_csv(output_csv, index=False)
    logger.info(f"Final CSV written with {len(combined_df)} rows to {output_csv}")
    
    # Master CSV remains unchanged
    logger.info("Master CSV was NOT modified.")


if __name__ == "__main__":
    main()
