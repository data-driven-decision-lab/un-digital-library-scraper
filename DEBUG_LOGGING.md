# Debug Logging for UN Digital Library Scraper

The scraper now includes comprehensive debug logging to help track what's happening during the scraping process and identify why specific records might be missed.

## How to Enable Debug Logging

### Method 1: Environment Variable (Recommended)
```bash
# Enable debug logging for the entire scraping session
export LOG_LEVEL=DEBUG
python src/un_data_pipeline/scraper_pipeline.py
```

### Method 2: Programmatically
You can also enable debug logging from within the script:
```python
from src.un_data_pipeline.scraper_pipeline import enable_debug_logging
enable_debug_logging()
```

## What Debug Logging Shows

### 1. **HTML Parsing Details**
- Number of metadata rows found on each page
- Extraction of titles, resolutions, vote dates, and vote summaries
- Individual country vote processing
- Validation checks (whether record has title/resolution)

### 2. **Link Collection Process**
- Year-by-year processing with page numbers
- Number of links found on each page
- NEW vs DUPLICATE link classification for each link
- Pagination navigation (next button clicks)

### 3. **Record Processing**
- Successful page loads vs timeouts
- HTML content size retrieved
- Step-by-step field extraction
- Validation outcomes (why records are accepted/rejected)

### 4. **Batch Processing**
- Batch progress and statistics
- Individual link processing within batches
- Success/failure rates per batch

### 5. **Year Processing Logic**
- Stopping criteria evaluation
- Session resets and browser management
- Duplicate detection triggering

## Sample Debug Output

```
[DEBUG] normalize_link: Processing href: https://digitallibrary.un.org/record/4068178?ln=en
[DEBUG] normalize_link: Normalized https://digitallibrary.un.org/record/4068178?ln=en -> https://digitallibrary.un.org/record/4068178
[INFO] Processing record: 4068178 - https://digitallibrary.un.org/record/4068178
[DEBUG] Navigating to: https://digitallibrary.un.org/record/4068178
[DEBUG] Page loaded successfully for record 4068178
[DEBUG] Retrieved HTML content (52483 chars) for record 4068178
[DEBUG] Starting HTML vote data extraction
[DEBUG] Found 12 metadata rows
[DEBUG] Processing metadata row 1: 'Title'
[DEBUG] Extracted Title: 'The occupied Syrian Golan : resolution / adopted by the General Assembly'
[DEBUG] Processing metadata row 3: 'Resolution'
[DEBUG] Extracted Resolution: 'A/RES/79/90'
[DEBUG] Processing metadata row 9: 'Vote date'
[DEBUG] Extracted Vote date: '2024-12-04'
[DEBUG] Processing metadata row 8: 'Vote summary'
[DEBUG] Extracted Vote summary: 'Voting Summary Yes: 150 | No: 4 | Abstentions: 25 | Non-Voting: 14 | Total voting membership: 193'
[DEBUG] Processing metadata row 10: 'Vote'
[DEBUG] Processing Vote data with 195 lines
[DEBUG] Extracted 179 country votes
[DEBUG] Record 4068178 - Title: 'The occupied Syrian Golan : resolution / adopted by the General Assembly', Council: 'General Assembly'
[DEBUG] Record 4068178 - Resolution: 'A/RES/79/90'
[DEBUG] Record 4068178 - Vote date: '2024-12-04'
[DEBUG] Record 4068178 - Vote counts: {'YES': '150', 'NO': '4', 'ABSTAIN': '25', 'NO-VOTE': '14', 'TOTAL': '193'}
[DEBUG] Record 4068178 - Country vote breakdown: {'YES': 150, 'NO': 4, 'ABSTAIN': 25}
[DEBUG] Record 4068178 - Validation: Title=True, Resolution=True
[INFO] Record 4068178 - Successfully processed with 197 fields
```

## Troubleshooting Missing Records

When a record is missing, look for these patterns in the debug logs:

1. **Record never encountered**: No log entries for the record ID
   - Check if the record was on a page that was processed
   - Look for pagination issues or early stopping

2. **Record failed parsing**: Look for extraction failures
   - Check for HTML structure changes
   - Verify metadata row parsing

3. **Record failed validation**: Look for validation debug messages
   - Missing title AND resolution
   - HTML extraction returned empty data

4. **Page load failures**: Look for timeout messages
   - Network issues during scraping
   - Page-specific loading problems

## Log File Location

Debug logs are written to: `logs/un_scraper_tagger.log`

## Performance Note

Debug logging significantly increases log output and may slow down the scraping process. Use it only when troubleshooting specific issues, not for production runs.