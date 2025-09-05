#!/usr/bin/env python3
"""
Test script to directly access and process document 4068178
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
        logging.FileHandler("logs/test_specific_document.log")
    ]
)
logger = logging.getLogger(__name__)

def test_specific_document():
    """Test processing the specific document directly"""
    from un_data_pipeline.scraper_pipeline import (
        get_driver, process_resolution, normalize_link
    )
    
    logger.info("Starting specific document test")
    
    target_link = "https://digitallibrary.un.org/record/4068178?ln=en"
    
    # Initialize driver
    driver = get_driver()
    try:
        logger.info(f"Testing direct access to: {target_link}")
        
        # First, just check if we can access the page
        driver.get(target_link)
        time.sleep(3)
        
        # Check if page loaded
        page_title = driver.title
        logger.info(f"Page title: {page_title}")
        
        # Check if we can find the voting data section
        voting_elements = driver.find_elements("xpath", "//div[contains(@class, 'metadata-row')]")
        logger.info(f"Found {len(voting_elements)} metadata rows")
        
        # Let's examine the page source to see what's actually there
        page_source = driver.page_source
        logger.info(f"Page source length: {len(page_source)} characters")
        
        # Look for key indicators
        if "vote" in page_source.lower():
            logger.info("Found 'vote' in page source")
        if "resolution" in page_source.lower():
            logger.info("Found 'resolution' in page source")
        if "golan" in page_source.lower():
            logger.info("Found 'golan' in page source") 
        if "A/RES/79/90" in page_source:
            logger.info("Found 'A/RES/79/90' in page source")
        
        # Try to process with our existing function
        logger.info("Attempting to process with process_resolution function...")
        row_data = process_resolution(target_link, driver, '2024')
        
        if row_data:
            logger.info("SUCCESS! Document was processed successfully!")
            logger.info(f"Extracted data:")
            for key, value in row_data.items():
                if key in ['Title', 'Resolution', 'Date', 'Council']:
                    logger.info(f"  {key}: {value}")
                elif key.endswith('COUNT') or key == 'TOTAL VOTES':
                    logger.info(f"  {key}: {value}")
            
            # Check for country votes
            country_votes = {k: v for k, v in row_data.items() 
                           if k not in ['Link', 'token', 'Scrape_Year', 'Title', 'Resolution', 'Date', 'Council'] 
                           and not k.endswith('COUNT') and k != 'TOTAL VOTES'}
            logger.info(f"Found {len(country_votes)} country votes")
            if country_votes:
                sample_votes = list(country_votes.items())[:5]
                logger.info(f"Sample country votes: {sample_votes}")
            
            return True, row_data
        else:
            logger.error("FAILED! Document could not be processed")
            
            # Let's examine what's in the metadata rows
            for i, elem in enumerate(voting_elements[:10]):  # Check first 10
                try:
                    elem_text = elem.text.strip()
                    if elem_text:
                        logger.info(f"Metadata row {i+1}: {elem_text[:100]}...")
                except Exception as e:
                    logger.warning(f"Error reading metadata row {i+1}: {e}")
            
            return False, None
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return False, None
        
    finally:
        driver.quit()

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TESTING SPECIFIC DOCUMENT 4068178")
    logger.info("=" * 60)
    
    success, data = test_specific_document()
    
    logger.info("=" * 60)
    logger.info(f"TEST RESULT: {'SUCCESS' if success else 'FAILED'}")
    if success and data:
        logger.info(f"Document Title: {data.get('Title', 'N/A')}")
        logger.info(f"Resolution: {data.get('Resolution', 'N/A')}")
        logger.info(f"Date: {data.get('Date', 'N/A')}")
    logger.info("=" * 60)