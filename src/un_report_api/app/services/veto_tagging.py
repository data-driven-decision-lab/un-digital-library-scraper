"""
Veto Analysis Tagging Module

This module provides tagging functionality for veto analysis data,
adapted from the UN scraper pipeline for use in the Security Council analysis.
"""

import pandas as pd
import numpy as np
import re
import ast
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI, APIConnectionError, RateLimitError
import random
import time

# Import the classification and geo-hierarchy data
import sys
import os

# Add the src directory to the Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up from sc_analysis/api/ to sc_analysis/ (project root is sc_analysis/..)
sc_analysis_dir = os.path.dirname(current_dir)  # sc_analysis/
project_root = os.path.dirname(sc_analysis_dir)  # project root (un-digital-library-scraper-main (1)/)
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from un_data_pipeline.data_modules.un_classification import un_classification
from un_data_pipeline.data_modules.un_geo_hierarchy import geo_hierarchy
from un_data_pipeline.data_modules.iso2_country import iso2_country_code

# Set up logging
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 1000

# Pydantic models for API responses
class LocationClassifications(BaseModel):
    continent: Optional[str] = Field(None, description="Continent of the country")
    subregion: Optional[str] = Field(None, description="Subregion of the country")
    country: Optional[str] = Field(None, description="Country")

class ResolutionTarget(BaseModel):
    classifications: List[LocationClassifications] = Field(..., description="List of relevant classifications for this resolution")

class MainTagClassification(BaseModel):
    """Pydantic model for stage 1 classification (main tags)"""
    main_tags: List[str] = Field(default_factory=list, description="List of relevant main category tags")

class SubTag1Classification(BaseModel):
    """Pydantic model for stage 2 classification (subtags)"""
    subtag1s: List[str] = Field(default_factory=list, description="List of relevant subcategories for the main tag")

class SubTag2Classification(BaseModel):
    """Pydantic model for stage 3 classification (specific items)"""
    subtag2s: List[str] = Field(default_factory=list, description="List of relevant specific items for the subcategory")

def create_openai_client() -> OpenAI:
    """Create and return an OpenAI client instance using the API key."""
    import os
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for veto_tagging")
    return OpenAI(api_key=api_key, timeout=20.0, max_retries=0)

def execute_api_call(api_call_fn, max_retries=5):
    """
    Execute an API call with robust rate limit and connection error handling.
    Uses exponential backoff with jitter on retryable errors.
    """
    retries = 0
    client = create_openai_client()

    while retries < max_retries:
        try:
            return api_call_fn(client)
        except (RateLimitError, APIConnectionError) as e:
            logger.warning(f"API Error ({type(e).__name__}). Retrying... (Attempt {retries+1}/{max_retries})")
            
            # Exponential backoff with jitter
            wait_time = (2 ** retries) + random.uniform(0, 3)
            logger.warning(f"Waiting {wait_time:.2f} seconds before retrying...")
            time.sleep(wait_time)
            
            retries += 1
            continue
        except Exception as e:
            raise e

    logger.error("Max retries reached. Halting API calls for this request.")
    raise Exception("OpenAI API request failed after max retries")

def call_llm_api(title: str, geo_hierarchy: dict, model: str = DEFAULT_MODEL) -> ResolutionTarget:
    """
    Analyzes a UN resolution text using LLM to identify if it's related to specific countries or regions.
    """
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
        
        classification_result: ResolutionTarget = response.choices[0].message.parsed
        logger.info("API call successful.")
        return classification_result
        
    except Exception as e:
        logger.error(f"Error during API call: {e}")
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
    """
    logger.info(f"Getting LLM tags for: {title[:50]}...")
    classification_result = call_llm_api(title, geo_hierarchy, model)
    
    result = []
    for classification in classification_result.classifications:
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
                        use_llm: bool = True):
    """
    Tags countries, subregions, and continents in a dataframe using both pattern matching and LLM.
    """
    logger.info(f"DataFrame columns before geo-tagging: {df.columns.tolist()}")
    
    # Check for and remove any existing geographic columns
    for col in ['country', 'subregion', 'continent']:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
        
    logger.info(f"DataFrame columns after cleanup: {df.columns.tolist()}")
    
    # Determine insertion position (after "canonical_topic")
    if 'canonical_topic' in df.columns:
        res_idx = list(df.columns).index('canonical_topic')
        insert_pos = res_idx + 1
    else:
        insert_pos = 1 if len(df.columns) > 1 else len(df.columns)
    
    # Create empty lists for our new columns
    countries_list = [[] for _ in range(len(df))]
    subregions_list = [[] for _ in range(len(df))]
    continents_list = [[] for _ in range(len(df))]
    
    # Pre-process country patterns for more accurate matching
    all_countries = []
    all_subregions = []
    all_continents = []
    
    for continent in geo_hierarchy.keys():
        all_continents.append(continent)
        
        for subregion, countries_or_dict in geo_hierarchy[continent].items():
            all_subregions.append((subregion, continent))
            
            if isinstance(countries_or_dict, dict):
                for subsubregion, countries in countries_or_dict.items():
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
    
    for idx, row in df.iterrows():
        title = str(row['canonical_topic'])  # Use canonical_topic as the title
        
        # Find all country matches
        found_countries = False
        for country, subregion, continent in all_countries:
            if country_patterns[country].search(title):
                found_countries = True
                iso_code = iso2_country_code.get(country, country)
                if iso_code and iso_code not in countries_list[idx]:
                    countries_list[idx].append(iso_code)
                    if subregion not in subregions_list[idx]:
                        subregions_list[idx].append(subregion)
                    if continent not in continents_list[idx]:
                        continents_list[idx].append(continent)
        
        # Find all subregion matches (if not already handled by country)
        if not found_countries:
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
        
        for idx in range(len(df)):
            title = str(df.iloc[idx]['canonical_topic'])
            llm_results = get_llm_location_tags(title, geo_hierarchy, model)
            
            # Add any new results from LLM, avoiding duplicates
            for continent, subregion, country in llm_results:
                if country:
                    iso_code = iso2_country_code.get(country, country)
                    if iso_code and iso_code not in countries_list[idx]:
                        countries_list[idx].append(iso_code)
                
                if subregion and subregion not in subregions_list[idx]:
                    subregions_list[idx].append(subregion)
                    
                if continent and continent not in continents_list[idx]:
                    continents_list[idx].append(continent)
    
    # STEP 3: Convert lists to comma-separated strings with better duplicate handling
    for idx in range(len(df)):
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
    Get classification tags for a resolution title using UN classification hierarchy.
    Returns the string representation of the list-of-lists.
    """
    tags = get_tags_sequential(title, model=model)
    return str(tags)

def process_tags(tag_str):
    """
    Convert a string representation of a list-of-lists (from the tagging API)
    into a comma-separated string for the 'tags' column.
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

def tag_veto_data(df, geo_hierarchy, iso2_country_code, model=DEFAULT_MODEL):
    """
    Process veto data using both traditional tagging and geo-tagging.
    """
    logger.info(f"Tagging {len(df)} veto resolution titles...")

    # Step 1: Apply traditional tagging function to each canonical_topic
    df['tags_raw'] = df['canonical_topic'].apply(lambda t: tag_resolution(t, model=model))
    
    # Step 2: Process tags to extract non-geographic tags
    df['tags'] = df['tags_raw'].apply(process_tags)
    
    # Remove temporary column
    df.drop('tags_raw', axis=1, inplace=True)
    
    # Step 3: Apply geo-tagging to add country, subregion, continent columns
    df = combined_geo_tagger(
        df=df,
        geo_hierarchy=geo_hierarchy,
        iso2_country_code=iso2_country_code,
        model=model,
        use_llm=True
    )
    
    logger.info("Veto data tagging complete.")
    return df
