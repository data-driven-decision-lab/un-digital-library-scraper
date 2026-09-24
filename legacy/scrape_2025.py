#!/usr/bin/env python3
"""
Standalone script to scrape UN voting data for year 2025 only.
Saves results to a local CSV file instead of Supabase.
"""
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import re
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create required directories
os.makedirs("logs", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Import scraper functions from the pipeline
from un_data_pipeline.scraper_pipeline import (
    get_driver,
    get_available_years,
    select_year_facet,
    collect_links_for_year,
    batch_scrape_resolutions,
    parallel_scrape_resolutions,
    retry_failed_links,
    clear_filters,
    tag_new_rows,
    standardize_country_columns,
    DuplicateLinkFound,
    BASE_SEARCH_URL,
    MAX_WORKERS,
    DEFAULT_MODEL,
    geo_hierarchy,
    iso2_country_code,
    logger,
)

TARGET_YEAR = "2025"


def main():
    driver = None
    try:
        logger.info(f"Starting UN voting data scraper for year {TARGET_YEAR} (CSV output)...")

        # No Supabase — start with an empty set of existing links
        existing_links: set = set()

        # Check if a partial CSV already exists to avoid re-scraping
        output_csv = f"data/raw/un_votes_{TARGET_YEAR}.csv"
        if os.path.exists(output_csv):
            logger.info(f"Found existing CSV: {output_csv}. Loading links for deduplication...")
            existing_df = pd.read_csv(output_csv)
            if 'Link' in existing_df.columns:
                existing_links = set(existing_df['Link'].dropna().tolist())
                logger.info(f"Loaded {len(existing_links)} existing links from CSV.")

        # Initialize browser
        driver = get_driver()
        driver.get(BASE_SEARCH_URL)
        time.sleep(2)

        years_data = get_available_years(driver)
        if not years_data:
            logger.error("No years found on the page. Check the website structure.")
            return

        logger.info(f"Available years: {[y['year'] for y in years_data]}")

        # Find the 2025 entry
        year_2025 = None
        for yd in years_data:
            if yd['year'] == TARGET_YEAR:
                year_2025 = yd
                break

        if year_2025 is None:
            logger.error(f"Year {TARGET_YEAR} not found in available years!")
            return

        logger.info(f"Found year {TARGET_YEAR} with {year_2025['count']} records.")

        # Select the year facet
        success, driver = select_year_facet(driver, year_2025)
        if not success:
            logger.error(f"Failed to select facet for {TARGET_YEAR}.")
            return

        # Collect links
        try:
            new_links = collect_links_for_year(driver, TARGET_YEAR, existing_links)
            logger.info(f"Collected {len(new_links)} new links for {TARGET_YEAR}.")
        except DuplicateLinkFound as e:
            new_links = e.new_links
            logger.info(f"Duplicate link rule triggered; found {len(new_links)} new links.")

        if not new_links:
            logger.info("No new links found. Nothing to scrape.")
            return

        # Scrape resolution pages
        BATCH_SIZE = 80
        if len(new_links) > 50 and MAX_WORKERS > 1:
            logger.info(f"Using parallel scraping with {MAX_WORKERS} workers for {len(new_links)} links")
            batch_rows, failed_links = parallel_scrape_resolutions(new_links, TARGET_YEAR, MAX_WORKERS)
        else:
            logger.info(f"Using sequential scraping for {len(new_links)} links")
            batch_rows, failed_links = batch_scrape_resolutions(new_links, driver, TARGET_YEAR, BATCH_SIZE)

        logger.info(f"Scraping done: {len(batch_rows)} successful, {len(failed_links)} failed")

        # Retry failed links
        if failed_links:
            retry_rows = retry_failed_links(failed_links, TARGET_YEAR)
            if retry_rows:
                batch_rows.extend(retry_rows)
                logger.info(f"Recovered {len(retry_rows)} rows from retry. Total: {len(batch_rows)}")

        if not batch_rows:
            logger.info("No rows scraped. Exiting.")
            return

        # Build DataFrame
        new_df = pd.DataFrame(batch_rows)
        logger.info(f"Raw scraped DataFrame: {len(new_df)} rows, {len(new_df.columns)} columns")

        # Save raw data first (before tagging)
        raw_csv = f"data/raw/un_votes_{TARGET_YEAR}_raw.csv"
        new_df.to_csv(raw_csv, index=False)
        logger.info(f"Raw data saved to {raw_csv}")

        # Tag new rows (subject tags + geo tags via LLM)
        logger.info("Applying tagging (subject + geo) to new rows...")
        tagged_df = tag_new_rows(
            new_df,
            geo_hierarchy=geo_hierarchy,
            iso2_country_code=iso2_country_code,
            model=DEFAULT_MODEL,
            max_workers=1,
        )

        # Standardize country columns
        logger.info("Standardizing country columns to ISO3...")
        final_df = standardize_country_columns(tagged_df)

        # Sort by date
        final_df['Date'] = pd.to_datetime(final_df['Date'], errors='coerce')
        final_df.sort_values('Date', ascending=True, inplace=True)
        final_df.reset_index(drop=True, inplace=True)

        # Save final processed CSV
        processed_csv = f"data/processed/un_votes_{TARGET_YEAR}_processed.csv"
        final_df.to_csv(processed_csv, index=False)
        logger.info(f"Processed data saved to {processed_csv}")

        # Also save/append to the running raw CSV for deduplication on next run
        if os.path.exists(output_csv):
            prev_df = pd.read_csv(output_csv)
            combined = pd.concat([prev_df, new_df], ignore_index=True)
            combined.drop_duplicates(subset=['Link'], keep='first', inplace=True)
            combined.to_csv(output_csv, index=False)
            logger.info(f"Appended to {output_csv}. Total rows: {len(combined)}")
        else:
            new_df.to_csv(output_csv, index=False)
            logger.info(f"Created {output_csv} with {len(new_df)} rows.")

        logger.info(f"\n{'='*60}")
        logger.info(f"DONE — {len(final_df)} records for {TARGET_YEAR}")
        logger.info(f"  Raw:       {raw_csv}")
        logger.info(f"  Processed: {processed_csv}")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        if driver:
            driver.quit()


if __name__ == "__main__":
    main()
