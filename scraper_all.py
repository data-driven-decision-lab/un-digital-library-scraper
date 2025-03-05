import os
import time
import re
import json
import csv
import logging
import platform
import pandas as pd
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

# Configure logging for verbose output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

# Constants
BASE_SEARCH_URL = (
    "https://digitallibrary.un.org/search?cc=Voting%20Data&ln=en&p=&f=&rm=&sf=&so=d"
    "&rg=50&c=Voting%20Data&c=&of=hb&fti=1&fct__9=Vote&fti=1"
)
CSV_FILE = "UN_DATA_FIXED.csv"
MAX_PAGES_PER_YEAR = 50  # Adjust as needed

# Fixed columns for CSV output
FIXED_COLUMNS = [
    "Council", "Date", "Title", "Resolution", "TOTAL VOTES", "NO-VOTE COUNT",
    "ABSENT COUNT", "NO COUNT", "YES COUNT", "Link", "token"
]

def prevent_sleep():
    """Prevent system sleep if running on Windows. (macOS users typically don't require this.)"""
    if platform.system() == 'Windows':
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception as e:
            logging.warning(f"Error preventing sleep: {e}")

def get_driver():
    """Configure and return a Chrome webdriver instance using webdriver_manager for macOS."""
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    # Get driver path using webdriver_manager
    driver_path = ChromeDriverManager().install()
    try:
        os.chmod(driver_path, 0o755)
    except Exception as e:
        logging.warning(f"Could not set executable permissions for {driver_path}: {e}")
        
    service = Service(executable_path=driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_page_load_timeout(60)
    return driver

def check_for_next_button(driver):
    """Return the next page button element if found, otherwise None."""
    next_button = None
    xpaths = [
        "//a[.//img[@alt='next'] or .//img[@aria-label='Next page']]",
        "//a[@class='img' and contains(@href, 'jrec=')]",
        "//a[contains(text(), '›') or contains(text(), 'next') or contains(text(), 'Next')]"
    ]
    for xp in xpaths:
        try:
            next_button = driver.find_element(By.XPATH, xp)
            if next_button:
                break
        except NoSuchElementException:
            continue
    if not next_button:
        pagination_links = driver.find_elements(By.XPATH, "//a[contains(@href, 'jrec=')]")
        highest_jrec = 0
        for link in pagination_links:
            try:
                href = link.get_attribute("href")
                match = re.search(r'jrec=(\d+)', href)
                if match:
                    jrec = int(match.group(1))
                    if jrec > highest_jrec:
                        highest_jrec = jrec
                        next_button = link
            except Exception:
                continue
    return next_button

def get_available_years(driver):
    """Extract available years and their counts from the date facet."""
    date_facets = []
    try:
        WebDriverWait(driver, 20).until(
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
                        WebDriverWait(driver, 10).until(
                            lambda d: "expanded" in facet_section.get_attribute("class")
                        )
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
                logging.error(f"Error processing date header: {e}")
                continue
        return sorted(date_facets, key=lambda x: x['year'], reverse=True)
    except Exception as e:
        logging.error(f"Error getting available years: {e}")
        return []

def select_year_facet(driver, year_data):
    """Select a specific year in the date facet and wait for results to refresh."""
    try:
        logging.info(f"Selecting year: {year_data['year']} ({year_data['count']} records)")
        checkbox = driver.find_element(By.ID, year_data['input_id'])
        driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, year_data['input_id'])))
        driver.execute_script("arguments[0].click();", checkbox)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/record/')]"))
        )
        logging.info(f"Filter applied for year {year_data['year']}")
        return True
    except TimeoutException:
        logging.error(f"Timeout waiting for results after selecting year {year_data['year']}")
        return False
    except Exception as e:
        logging.error(f"Error selecting year facet {year_data['year']}: {e}")
        return False

def collect_links_for_year(driver, year):
    """Collect all unique record links for the given year."""
    all_links = set()
    page_count = 0
    consecutive_no_new = 0
    max_no_new = 3

    while page_count < MAX_PAGES_PER_YEAR:
        prevent_sleep()
        page_count += 1
        logging.info(f"[Year {year}] Processing page {page_count}...")
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/record/')]"))
            )
        except TimeoutException:
            logging.warning(f"[Year {year}] Timeout waiting for results on page {page_count}")
            break

        links_before = len(all_links)
        try:
            for elem in driver.find_elements(By.XPATH, "//a[contains(@href, '/record/')]"):
                try:
                    href = elem.get_attribute("href")
                    if href and '/record/' in href:
                        base_url = href.split('?')[0]
                        if '?' in href:
                            params = href.split('?')[1].split('&')
                            ln_param = [p for p in params if p.startswith('ln=')]
                            if ln_param:
                                base_url = f"{base_url}?{ln_param[0]}"
                        all_links.add(base_url)
                except StaleElementReferenceException:
                    continue
        except Exception as e:
            logging.error(f"[Year {year}] Error collecting direct links: {e}")

        try:
            for item in driver.find_elements(By.XPATH, "//div[contains(@class, 'searchresult')]"):
                try:
                    link_elem = item.find_element(By.XPATH, ".//a[contains(@href, '/record/')]")
                    href = link_elem.get_attribute("href")
                    if href and '/record/' in href:
                        all_links.add(href)
                except (NoSuchElementException, StaleElementReferenceException):
                    continue
        except Exception as e:
            logging.error(f"[Year {year}] Error collecting links from results: {e}")

        new_count = len(all_links) - links_before
        logging.info(f"[Year {year}] Found {new_count} new links on page {page_count} (Total: {len(all_links)})")
        if new_count == 0:
            consecutive_no_new += 1
            if consecutive_no_new >= max_no_new:
                logging.info(f"[Year {year}] No new links on {max_no_new} consecutive pages; stopping pagination.")
                break
        else:
            consecutive_no_new = 0

        next_button = check_for_next_button(driver)
        if next_button:
            try:
                driver.execute_script("arguments[0].scrollIntoView(true);", next_button)
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, ".//*")))
                driver.execute_script("arguments[0].click();", next_button)
                WebDriverWait(driver, 20).until(
                    EC.staleness_of(next_button)
                )
            except Exception as e:
                logging.error(f"[Year {year}] Error clicking next button: {e}")
                break
        else:
            logging.info(f"[Year {year}] No next button found; reached last page.")
            break
    return list(all_links)

def scrape_resolution(link, driver, retries=3):
    """
    Load a resolution page with retry logic.
    Returns the page's HTML content if successful, or None.
    """
    attempt = 0
    while attempt < retries:
        try:
            logging.info(f"Loading page: {link} (Attempt {attempt+1}/{retries})")
            driver.get(link)
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'metadata-row')]"))
            )
            time.sleep(0.5)
            return driver.page_source
        except TimeoutException as te:
            logging.warning(f"Timeout for {link}: {te}")
            attempt += 1
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error loading {link}: {e}")
            attempt += 1
            time.sleep(2)
    logging.error(f"Skipping {link} after {retries} failed attempts.")
    return None

def extract_vote_data_from_html(html_content):
    """Extract vote data from HTML using JSON-LD or scraping metadata rows."""
    soup = BeautifulSoup(html_content, "html.parser")
    data = {}
    try:
        script_tag = soup.find('script', {'type': 'application/ld+json', 'id': 'detailed-schema-org'})
        if script_tag and script_tag.string:
            json_data = json.loads(script_tag.string)
            data['Title'] = json_data.get('name', '')
            data['Date'] = json_data.get('datePublished', '')
    except Exception as e:
        logging.debug(f"JSON-LD extraction error: {e}")
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
        except Exception as e:
            logging.debug(f"Error parsing metadata row: {e}")
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

def save_to_csv(rows, filename=None, append=False):
    """Save collected data rows to a CSV file."""
    if not rows:
        logging.info("No data to save.")
        return False
    try:
        output_file = filename if filename else CSV_FILE
        all_cols = set(FIXED_COLUMNS)
        for row in rows:
            all_cols.update(row.keys())
        extra_cols = sorted(list(all_cols - set(FIXED_COLUMNS)))
        column_order = FIXED_COLUMNS + extra_cols
        mode = "a" if append else "w"
        with open(output_file, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=column_order)
            if not append:
                writer.writeheader()
            writer.writerows(rows)
        logging.info(f"Wrote {len(rows)} rows to {output_file} (size: {os.path.getsize(output_file)} bytes)")
        return True
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        return False

def save_links_to_file(links, year=None):
    """Save links to a text file as backup."""
    filename = f"un_resolution_links_{year}.txt" if year else "un_resolution_links_all.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            for link in links:
                f.write(f"{link}\n")
        logging.info(f"Saved {len(links)} links to {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving links: {e}")
        return False

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
                    WebDriverWait(driver, 10).until(EC.staleness_of(button))
                    logging.info("Cleared filters using clear button.")
                    return True
                except Exception:
                    continue
        driver.get(BASE_SEARCH_URL)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//a[contains(@href, '/record/')]"))
        )
        logging.info("Cleared filters by reloading the base page.")
        return True
    except Exception as e:
        logging.error(f"Error clearing filters: {e}")
        return False

def main():
    """Main function to run the scraper year by year."""
    logging.info("Starting UN Resolution Vote Scraper (Verbose & Optimized for macOS)")
    driver = get_driver()
    try:
        prevent_sleep()  # Will do nothing on macOS
        logging.info(f"Loading base search page: {BASE_SEARCH_URL}")
        driver.get(BASE_SEARCH_URL)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//ul[contains(@class, 'option-fct')]"))
        )
        logging.info("Base page loaded.")
        years_data = get_available_years(driver)
        if not years_data:
            logging.error("No year facets found; exiting.")
            return
        logging.info(f"Found {len(years_data)} years to process: {', '.join([y['year'] for y in years_data])}")

        master_links = []
        master_rows = []
        processed_years_file = "processed_years.txt"
        processed_years = set()
        if os.path.exists(processed_years_file):
            with open(processed_years_file, "r") as f:
                processed_years = set(line.strip() for line in f)
            logging.info(f"Previously processed years: {', '.join(processed_years)}")

        for year_data in years_data:
            year = year_data['year']
            if year in processed_years:
                logging.info(f"Skipping already processed year: {year}")
                continue
            logging.info(f"\n{'='*60}\nProcessing year {year} ({year_data['count']} records)")
            clear_filters(driver)
            if driver.current_url != BASE_SEARCH_URL:
                driver.get(BASE_SEARCH_URL)
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, "//ul[contains(@class, 'option-fct')]"))
                )
            if not select_year_facet(driver, year_data):
                logging.error(f"Failed to select facet for {year}; skipping.")
                continue

            year_links = collect_links_for_year(driver, year)
            if not year_links:
                logging.warning(f"No links found for year {year}.")
                continue
            logging.info(f"Collected {len(year_links)} links for year {year}")
            save_links_to_file(year_links, year)
            master_links.extend(year_links)

            year_rows = []
            for idx, link in enumerate(year_links, start=1):
                prevent_sleep()
                html = scrape_resolution(link, driver)
                if html is None:
                    continue
                record_id = link.split('/record/')[1].split('?')[0] if '/record/' in link else link.split('/')[-1]
                row_data = {"Link": link, "token": record_id}
                extracted = extract_vote_data_from_html(html)
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
                    year_rows.append(row_data)
                if idx % 20 == 0:
                    logging.info(f"[Year {year}] Processed {idx}/{len(year_links)} links. Saving batch...")
                    save_to_csv(year_rows, filename=f"UN_DATA_{year}.csv")
                    master_rows.extend(year_rows)
                    save_to_csv(master_rows, append=True)
                    year_rows = []
            if year_rows:
                logging.info(f"[Year {year}] Saving final batch of {len(year_rows)} rows.")
                save_to_csv(year_rows, filename=f"UN_DATA_{year}.csv")
                master_rows.extend(year_rows)
                save_to_csv(master_rows, append=True)
            with open(processed_years_file, "a") as f:
                f.write(f"{year}\n")
            processed_years.add(year)
            logging.info(f"Completed processing for year {year}")

        if master_rows:
            logging.info(f"Final save: {len(master_rows)} rows collected overall.")
            save_to_csv(master_rows)
        if master_links:
            save_links_to_file(master_links)
        logging.info("Data collection complete!")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        driver.quit()
        logging.info("Scraper finished.")

if __name__ == "__main__":
    main()