#!/usr/bin/env python3
"""
Test script to scrape only 2024 data and check for missing resolution A/RES/79/90
"""
import sys
import os
import logging
import time
from datetime import datetime

# Set dummy environment variables to avoid errors
os.environ['API_KEY'] = 'dummy_key_for_testing'

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/test_2024_scraper.log")
    ]
)
logger = logging.getLogger(__name__)

def test_2024_scraper():
    """Test scraper specifically for 2024 data"""
    from un_data_pipeline.scraper_pipeline import (
        get_driver, get_available_years, select_year_facet, 
        collect_links_for_year, process_resolution, normalize_link,
        BASE_SEARCH_URL
    )
    
    logger.info("Starting 2024 test scraper")
    
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
        
        # Select 2024
        success, driver = select_year_facet(driver, year_2024)
        if not success:
            logger.error("Failed to select 2024 year facet")
            return
        
        # Collect links for 2024 (using empty existing_links to get all)
        existing_links = set()
        new_links = collect_links_for_year(driver, '2024', existing_links)
        
        logger.info(f"Collected {len(new_links)} links for 2024")
        
        # Look for the specific missing record
        target_link = "https://digitallibrary.un.org/record/4068178"
        found_target = False
        
        for link in new_links:
            normalized = normalize_link(link)
            if target_link in normalized or "4068178" in normalized:
                logger.info(f"FOUND TARGET RECORD: {normalized}")
                found_target = True
                break
        
        if not found_target:
            logger.warning(f"Target record {target_link} NOT FOUND in collected links")
        
        # Process first few links to test functionality
        test_links = new_links[:5]
        logger.info(f"Testing processing of first {len(test_links)} links")
        
        successful_records = []
        for i, link in enumerate(test_links):
            logger.info(f"Processing test link {i+1}/{len(test_links)}: {link}")
            row_data = process_resolution(link, driver, '2024')
            if row_data:
                successful_records.append(row_data)
                logger.info(f"Successfully processed: {row_data.get('Resolution', 'N/A')} - {row_data.get('Title', 'N/A')[:50]}...")
            else:
                logger.warning(f"Failed to process: {link}")
        
        logger.info(f"Test completed: {len(successful_records)}/{len(test_links)} records processed successfully")
        
        # If we found the target, process it specifically
        if found_target:
            logger.info("Processing target record specifically...")
            target_data = process_resolution(target_link, driver, '2024')
            if target_data:
                logger.info(f"TARGET RECORD PROCESSED SUCCESSFULLY!")
                logger.info(f"Resolution: {target_data.get('Resolution', 'N/A')}")
                logger.info(f"Title: {target_data.get('Title', 'N/A')}")
                logger.info(f"Date: {target_data.get('Date', 'N/A')}")
                logger.info(f"Fields: {len(target_data)}")
            else:
                logger.error("TARGET RECORD FAILED TO PROCESS")
        
        # Check if any records have A/RES/79/90
        for record in successful_records:
            if record.get('Resolution') == 'A/RES/79/90':
                logger.info(f"FOUND A/RES/79/90 in processed records!")
                logger.info(f"Title: {record.get('Title', 'N/A')}")
                logger.info(f"Date: {record.get('Date', 'N/A')}")
        
        return {
            'total_links': len(new_links),
            'found_target': found_target,
            'successful_processing': len(successful_records),
            'test_records': successful_records
        }
        
    finally:
        driver.quit()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("STARTING 2024 TEST SCRAPER")
    logger.info("=" * 60)
    
    result = test_2024_scraper()
    
    if result:
        logger.info("=" * 60)
        logger.info("TEST RESULTS SUMMARY:")
        logger.info(f"Total 2024 links found: {result['total_links']}")
        logger.info(f"Target record found: {result['found_target']}")
        logger.info(f"Successful test processing: {result['successful_processing']}")
        logger.info("=" * 60)
    else:
        logger.error("Test failed")