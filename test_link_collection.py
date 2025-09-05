#!/usr/bin/env python3
"""
Test script to examine the link collection behavior for 2024
"""
import sys
import os
import logging
import time

# Set dummy environment variables to avoid errors
os.environ['API_KEY'] = 'dummy_key_for_testing'

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/test_link_collection.log")
    ]
)
logger = logging.getLogger(__name__)

def test_link_collection_2024():
    """Test link collection for 2024 with different scenarios"""
    from un_data_pipeline.scraper_pipeline import (
        get_driver, get_available_years, select_year_facet, 
        collect_links_for_year, get_links_from_csv_regex,
        BASE_SEARCH_URL, MASTER_CSV, DuplicateLinkFound
    )
    
    logger.info("Testing link collection behavior for 2024")
    
    # Initialize driver
    driver = get_driver()
    try:
        driver.get(BASE_SEARCH_URL)
        time.sleep(2)
        
        # Get available years
        years_data = get_available_years(driver)
        logger.info(f"Found {len(years_data)} available years")
        
        # Find 2024
        year_2024 = None
        for year_data in years_data:
            if year_data['year'] == '2024':
                year_2024 = year_data
                break
        
        if not year_2024:
            logger.error("Year 2024 not found in available years")
            return
        
        logger.info(f"Found 2024 data: {year_2024}")
        
        # Test 1: Collection with no existing links (should get everything)
        logger.info("=" * 50)
        logger.info("TEST 1: Collecting with no existing links")
        logger.info("=" * 50)
        
        success, driver = select_year_facet(driver, year_2024)
        if not success:
            logger.error("Failed to select 2024 year facet")
            return
        
        # Collect with empty existing_links
        try:
            new_links_no_existing = collect_links_for_year(driver, '2024', set())
            logger.info(f"Collected {len(new_links_no_existing)} links with no existing links")
            
            # Check if our target is there
            target_found = any("4068178" in link for link in new_links_no_existing)
            logger.info(f"Target 4068178 found: {target_found}")
            
            if target_found:
                target_link = [link for link in new_links_no_existing if "4068178" in link][0]
                logger.info(f"Target link: {target_link}")
        except DuplicateLinkFound as e:
            logger.info(f"DuplicateLinkFound with no existing links: {e.message}")
            new_links_no_existing = e.new_links
            logger.info(f"Got {len(new_links_no_existing)} links before stopping")
            
            target_found = any("4068178" in link for link in new_links_no_existing)
            logger.info(f"Target 4068178 found: {target_found}")
        
        # Test 2: Collection with existing CSV links (realistic scenario)
        logger.info("=" * 50)
        logger.info("TEST 2: Collecting with existing CSV links")
        logger.info("=" * 50)
        
        existing_links_csv = set(get_links_from_csv_regex(MASTER_CSV))
        logger.info(f"Loaded {len(existing_links_csv)} existing links from CSV")
        
        # Reset to 2024
        driver.get(BASE_SEARCH_URL)
        time.sleep(2)
        success, driver = select_year_facet(driver, year_2024)
        if not success:
            logger.error("Failed to select 2024 year facet")
            return
        
        try:
            new_links_with_existing = collect_links_for_year(driver, '2024', existing_links_csv)
            logger.info(f"Collected {len(new_links_with_existing)} new links with existing CSV links")
            
            target_found = any("4068178" in link for link in new_links_with_existing)
            logger.info(f"Target 4068178 found: {target_found}")
            
        except DuplicateLinkFound as e:
            logger.info(f"DuplicateLinkFound with existing CSV links: {e.message}")
            new_links_with_existing = e.new_links
            logger.info(f"Got {len(new_links_with_existing)} links before duplicate found")
            
            target_found = any("4068178" in link for link in new_links_with_existing)
            logger.info(f"Target 4068178 found in partial collection: {target_found}")
            
            if not target_found:
                logger.warning("TARGET NOT FOUND - This explains why the scraper is missing it!")
                logger.info("The scraper stops due to duplicate detection before reaching our target document")
        
        # Summary
        logger.info("=" * 50)
        logger.info("SUMMARY")
        logger.info("=" * 50)
        logger.info("This test demonstrates whether the duplicate detection logic")
        logger.info("is preventing the scraper from finding document 4068178")
        
    finally:
        driver.quit()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    test_link_collection_2024()