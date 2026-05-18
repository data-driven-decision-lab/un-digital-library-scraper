# Architecture

**Analysis Date:** 2026-05-18

## Pattern Overview

**Overall:** Multi-stage ETL pipeline (Extract → Transform → Load) with REST API exposure.

**Key Characteristics:**
- Three independent but coordinated pipelines operating on shared Turso database
- Staged tagging approach (raw scrape → classification → geo-tagging → upload)
- Fallback to HTTP-based Turso client for Windows/CI compatibility
- Dual storage model: raw JSON vote data + denormalized country columns
- CSV export as database fallback for API service

## Layers

**Scraper Layer:**
- Purpose: Extract UN voting data from digitallibrary.un.org via Selenium, perform initial classification
- Location: `src/un_data_pipeline/scraper_pipeline.py`
- Contains: Selenium web automation, Gemini LLM API calls, data cleaning, deduplication logic
- Depends on: Selenium, BeautifulSoup, OpenAI/Gemini SDK, pycountry, Turso database
- Used by: Manual execution via `python -c "import sys; sys.path.insert(0, 'src'); from un_data_pipeline.scraper_pipeline import main; main()"`

**Enrichment Layer:**
- Purpose: Enrich raw voting records with subject tags and geo-classifications
- Location: `src/un_data_pipeline/scraper_pipeline.py` (functions `get_tags_sequential`, `combined_geo_tagger`, `tag_new_rows`)
- Contains: LLM-based tagging, regex-based country matching, hierarchical geo-classification
- Depends on: `data_modules/un_classification.py`, `data_modules/un_geo_hierarchy.py`, `data_modules/iso2_country.py`
- Used by: Main scraper after link collection and resolution parsing

**Dashboard Pipeline Layer:**
- Purpose: Aggregate voting data, compute pillar scores, derive rankings and similarity metrics
- Location: `src/un_data_pipeline/dashboard_data_pipeline.py`
- Contains: Score calculation, year-over-year aggregation, cosine similarity matrix generation, regional mapping
- Depends on: pandas, scikit-learn, Turso database
- Used by: Scheduled pipeline execution or manual trigger via `python -m src.un_data_pipeline.dashboard_data_pipeline`

**API Layer:**
- Purpose: Serve country reports, rankings, and Security Council analysis
- Location: `src/un_report_api/app/main.py`
- Contains: FastAPI endpoints, request validation, response generation, CORS middleware
- Depends on: FastAPI, pandas, data_loader service, ranking generator
- Used by: HTTP clients via `uvicorn main:app --host 0.0.0.0 --port 8000`

**Database Abstraction Layer:**
- Purpose: Provide Turso connectivity with platform-specific fallback
- Location: `src/un_data_pipeline/turso_http.py` (HTTP client), libsql-experimental (native client)
- Contains: TursoHTTPConnection class mimicking DB-API 2.0 cursor, batched request handling, type conversion
- Depends on: libsql-experimental (Linux/macOS) or urllib (Windows fallback)
- Used by: All pipeline stages for read/write operations

## Data Flow

**Scraper → Raw Table:**

1. Scraper loads existing links from `un_votes_raw` / `un_votes_with_sc` for deduplication
2. Selenium navigates digitallibrary.un.org, collects year-wise resolution links (BASE_SEARCH_URL)
3. For each new link, `process_resolution()` parses vote grid, extracts per-country voting data
4. Raw rows written to `un_votes_raw` table with vote_data as JSON blob
5. Checkpoint progress after each year to `pipeline_runs` table
6. Full dataset enriched with tags and sc_flag, uploaded to `un_votes_with_sc`

**Enrichment Process:**

- Tags extracted via staged LLM API calls: MainTagClassification → SubTag1Classification → SubTag2Classification
- Geo-tagging combines regex matching (country names via iso2_country_code) with LLM analysis (geo_hierarchy)
- `combined_geo_tagger()` produces country, subregion, continent columns
- Results standardized to ISO-3 codes before upload

**Dashboard Pipeline:**

1. Load all rows from `un_votes_with_sc`
2. Filter out Security Council resolutions (Resolution starts with 'S/')
3. Expand vote_data JSON → per-country boolean columns
4. Compute per-country pillar scores annually:
   - Pillar 1: Alignment with world (cosine similarity to mean vote vector)
   - Pillar 2: Regional consensus (alignment with region median)
   - Pillar 3: Topic leadership (vote % on specific topics)
5. Normalize scores to 0-100 scale, compute rankings
6. Generate pairwise cosine similarity matrix for all country pairs per year
7. Aggregate topic-wise vote counts (yes/no/abstain) per country per year
8. Write to `annual_scores`, `topic_votes_yearly`, `pairwise_similarity_yearly` tables
9. Export as CSV to `src/un_report_api/app/required_csvs/` for API fallback

**API Request → Response:**

1. Client requests `/report/{country_iso}?start_year=X&end_year=Y`
2. Ranking generator loads annual_scores CSV/Turso
3. Report generator loads pillar scores, similarity matrix, topic votes
4. Assembles country report: scores, rankings, allies/enemies, topic analysis, regional context
5. Returns ReportResponse JSON with all computed metrics

**State Management:**

- Single source of truth: Turso database tables
- Scraper logs execution metadata to `pipeline_runs` (run_id, status, rows_affected, error_message)
- CSV files serve as cache/fallback when database unavailable
- No in-memory state across pipeline runs; each stage is independently idempotent

## Key Abstractions

**Resolution (Scraped Record):**
- Purpose: Represents a single UN General Assembly voting session
- Examples: `scraper_pipeline.py` lines ~1368 (process_resolution), ~2069 (process_and_upload_data)
- Pattern: Dictionary with keys: Resolution (e.g., 'A/77/RES/1'), Date, Title, Link (unique), tags (CSV), vote_data (JSON)
- vote_data structure: `{"USA": "YES", "CHN": "NO", "RUS": "ABSTAIN", ...}` (per-country votes)

**LocationClassifications (LLM Output):**
- Purpose: Represents geographic metadata extracted via Gemini for a single resolution
- Examples: `scraper_pipeline.py` lines ~288 (Pydantic model)
- Pattern: Structured output from LLM with continent, subregion, country fields (optional)
- Used for: Enriching resolution records with geo-tags before upload

**AnnualScore (Computed Result):**
- Purpose: Represents a single country's performance in a given year
- Examples: `dashboard_data_pipeline.py` lines ~623 (generate_annual_scores)
- Pattern: DataFrame row with columns: Year, Country, Pillar 1/2/3 Score, Total Index, ranks (per-country and overall)
- Used for: Rankings generation and country report pillar metrics

**PairwiseSimilarity (Relationship Metric):**
- Purpose: Represents cosine similarity between two countries' voting patterns in a year
- Examples: `dashboard_data_pipeline.py` lines ~783 (generate_similarity_matrix)
- Pattern: Row with Year, Country1, Country2, CosineSimilarity (0-1 range)
- Used for: Finding top allies/enemies in country reports

## Entry Points

**Scraper Pipeline:**
- Location: `src/un_data_pipeline/scraper_pipeline.py` line 2178 (main function)
- Triggers: Manual execution via `python -c "...from un_data_pipeline.scraper_pipeline import main; main()"`
- Responsibilities: 
  - Initialize Selenium driver, load UN Digital Library search page
  - Iterate through available years (2023-1946), collect resolution links
  - Dedup against existing records in Turso
  - Scrape each resolution's vote grid, extract per-country vote data
  - Classify new rows with UNBIS tags and geo-tags via Gemini LLM
  - Upload to `un_votes_raw` (raw scrape) and `un_votes_with_sc` (enriched)
  - Log execution metadata to `pipeline_runs`

**Dashboard Pipeline:**
- Location: `src/un_data_pipeline/dashboard_data_pipeline.py` line 852 (main function)
- Triggers: Manual execution via `python -m src.un_data_pipeline.dashboard_data_pipeline`
- Responsibilities:
  - Load all rows from `un_votes_with_sc` table
  - Compute pillar scores, rankings, topic aggregates, similarity matrix
  - Save results to Turso tables and CSV files
  - Record pipeline execution metadata

**API Application:**
- Location: `src/un_report_api/app/main.py` (FastAPI app instance)
- Triggers: `uvicorn main:app --host 0.0.0.0 --port 8000` or Cloud Run deployment
- Responsibilities:
  - Expose endpoints: `/report/{country_iso}`, `/rankings/{year}`, `/sc/veto_analysis`, `/sc/vote_analysis`
  - Load data from Turso or CSV fallback
  - Generate country reports with pillar scores, rankings, regional analysis
  - Serve Security Council analysis (veto patterns, vote distribution)

## Error Handling

**Strategy:** Structured logging with fallback mechanisms; graceful degradation when external services unavailable.

**Patterns:**

- **Scraper API failures:** `execute_api_call()` implements exponential backoff with jitter (max 5 retries) for RateLimitError and APIConnectionError from Gemini API
- **Selenium browser crashes:** Driver creation wrapped in try/except; on failure, new driver spawned and search page reloaded
- **Database connection:** Automatic fallback from libsql-experimental (Unix) to HTTP-based client (Windows) via `_USE_HTTP_CLIENT` flag
- **Data validation:** `validate_source_year_coverage()` and `validate_output_contains_year()` ensure required years present in pipeline outputs
- **Logging:** All stages log to file (`logs/un_scraper_tagger.log`) and console; log level configurable via `LOG_LEVEL` environment variable (default INFO)
- **Pipeline metadata:** Run status (running/success/failed) recorded in `pipeline_runs` table with error_message field for diagnostics

## Cross-Cutting Concerns

**Logging:** Configured at module initialization with StreamHandler (console) and FileHandler (`logs/un_scraper_tagger.log`). Scraper uses module-level logger; dashboard and API use their own loggers.

**Validation:** Pydantic models enforce type validation (LocationClassifications, ResolutionTarget). Dataframe validation functions check for required columns and year coverage.

**Authentication:** OpenAI/Gemini API key loaded via `os.getenv('API_KEY')`. Turso credentials (TURSO_DATABASE_URL, TURSO_AUTH_TOKEN) loaded via environment. API endpoints have no authentication (public facing, CORS restricted to datadrivendecisionlab.com).

**Rate Limiting:** Gemini API calls wrapped in retry logic with exponential backoff; Turso HTTP batching limits 500 statements per batch; Selenium session reset after 150 requests to prevent browser staleness.

**Deduplication:** Link uniqueness enforced at Turso schema level (UNIQUE constraint on un_votes_raw.Link and un_votes_with_sc.Link). Scraper maintains in-memory set of existing links for fast lookup during collection.

---

*Architecture analysis: 2026-05-18*
