# Architecture

**Analysis Date:** 2026-03-18

## Pattern Overview

**Overall:** Layered Pipeline + REST API Architecture

The system is organized as two distinct subsystems that share data:
1. **Data Pipeline** - Batch processing layer that scrapes, classifies, and aggregates UN voting data
2. **REST API** - Query layer that serves pre-processed analytics and reports

**Key Characteristics:**
- Data flows one direction: raw web data → pipeline → Supabase database → API endpoints
- Separation between data collection/processing (scraper_pipeline) and serving (FastAPI)
- Security Council analysis is a specialized analytical layer built on pipeline outputs
- Local CSV files and Supabase serve as dual storage tiers for different data types

## Layers

**Web Scraping & Data Collection:**
- Purpose: Extract UN resolution voting data from UN Digital Library
- Location: `src/un_data_pipeline/scraper_pipeline.py`
- Contains: Selenium-based web scraper, incremental deduplication, raw data capture
- Depends on: BeautifulSoup, Selenium, webdriver-manager, Supabase
- Used by: Dashboard data pipeline (reads Supabase output)

**Classification & Tagging Layer:**
- Purpose: Enrich raw resolution data with subject matter and geographical tags
- Location: `src/un_data_pipeline/scraper_pipeline.py` (integrated into pipeline)
- Contains: OpenAI LLM calls for subject classification, pattern-based geo-tagging, ISO country mapping
- Depends on: `data_modules/un_classification.py`, `data_modules/un_geo_hierarchy.py`, OpenAI API
- Used by: Dashboard pipeline for enriched data

**Data Aggregation & Scoring:**
- Purpose: Calculate pillar scores, rankings, and similarity metrics from voting records
- Location: `src/un_data_pipeline/dashboard_data_pipeline.py`
- Contains: Pandas-based aggregation, score calculation from voting patterns, cosine similarity computation
- Depends on: Supabase (source), scikit-learn for similarity, pandas for transformation
- Used by: Report API, rankings endpoints

**API Request Handling Layer:**
- Purpose: Expose analytical data via REST endpoints
- Location: `src/un_report_api/app/main.py`
- Contains: FastAPI application, endpoint routing, request validation, CORS middleware
- Depends on: FastAPI, Pydantic models, logging middleware
- Used by: Frontend clients (datadrivendecisionlab.com)

**Report Generation Layer:**
- Purpose: Synthesize analytics into structured country and regional reports
- Location: `src/un_report_api/app/report_generator.py`
- Contains: Multi-section report assembly, alignment scoring (P5 similarity), topic analysis
- Depends on: CSV data loaders, `country_iso_map.py`, Supabase (optional fallback)
- Used by: `/report/{country_iso}` endpoint

**Rankings & Aggregation Layer:**
- Purpose: Compute annual pillar rankings across all countries
- Location: `src/un_report_api/app/ranking_generator.py`
- Contains: Ranking calculation (higher score = better rank), year-over-year change computation, Pillar 1/2/3 stratification
- Depends on: `annual_scores.csv`, pandas
- Used by: `/rankings/{year}` endpoint

**Security Council Analysis Layer:**
- Purpose: Analyze P5 veto patterns, consensus behavior, and voting alignment
- Location: `src/un_report_api/app/services/` (analysis_service.py, vote_analysis_service.py)
- Contains: Veto tagging, behavioral analysis, voting pattern extraction
- Depends on: `sc_data/` CSV files, pandas, numpy
- Used by: `/sc/veto_analysis`, `/sc/vote_analysis` endpoints

**Data Access Layer:**
- Purpose: Abstract database and file system access
- Location: `src/un_report_api/app/supabase_client.py` and `src/un_report_api/app/services/data_loader.py`
- Contains: Supabase client initialization, CSV loading utilities, path resolution
- Depends on: Supabase SDK, pandas, os/logging
- Used by: Pipeline and API layers for data retrieval

## Data Flow

**Scraping → Tagging → Storage:**

1. `scraper_pipeline.py` main() loads existing master CSV
2. Selenium driver navigates UN Digital Library, extracts new resolutions
3. BeautifulSoup parses resolution titles, voting records
4. Deduplication via token hashing prevents re-processing
5. OpenAI classifies subjects (tags) and performs geo-tagging (country/subregion/continent)
6. Rows merged with existing data, sorted by date, re-indexed
7. Final CSV written with timestamp suffix (e.g., UN_VOTING_DATA_RAW_WITH_TAGS_2025-03-18.csv)
8. Data uploaded to Supabase `un_votes_with_sc` table

**Aggregation → Scoring:**

1. `dashboard_data_pipeline.py` reads from Supabase `un_votes_with_sc` table
2. Computes voting percentages (YES/NO/ABSTAIN) by country, topic, region
3. Calculates three "pillars" (scoring dimensions) per country per year
4. Computes cosine similarity between all country pairs (pairwise_similarity_yearly.csv)
5. Generates rankings by pillar
6. Writes outputs: annual_scores.csv, topic_votes_yearly.csv, country/regional breakdowns
7. Outputs staged at `data/processed/` (local) and project root (for API)

**API Query → Report Assembly:**

1. FastAPI endpoint `/report/{country_iso}` receives GET request
2. `validate_year_params()` validates start_year/end_year bounds
3. `report_generator.generate_report()` loads CSVs and Supabase data
4. Computes report sections:
   - Metadata (country name, period)
   - Pillar scores and ranks (period averages)
   - Voting behavior (vote percentages vs. world average)
   - P5 alignment (most/least aligned via cosine similarity)
   - Top allies/enemies (countries with highest/lowest similarity)
   - Topic voting patterns
   - Regional peer alignment
5. Pydantic model validates structure (ReportResponse)
6. JSON serialized to client

**Security Council Analysis Flow:**

1. `/sc/veto_analysis` endpoint calls `get_enhanced_veto_analysis()`
2. `SecurityCouncilDataLoader` resolves available data files (enhanced_veto_descriptions.json → final_analysis_data.csv → fallbacks)
3. `SecurityCouncilAnalysisService` extracts veto records, power dynamics
4. Data tagged with canonical labels, deterministic statistics computed
5. JSON response with veto patterns, behavioral trends returned

**State Management:**

- **Pipeline State:** Tracks processed resolutions via token hashing; new items identified through set difference
- **Data State:** CSV files serve as source of truth (annual_scores.csv, topic_votes_yearly.csv)
- **Supabase:** Primary database for raw voting records; queried on-demand during report generation
- **Cache:** No in-memory caching; CSVs and Supabase provide persistent state

## Key Abstractions

**Country Identification:**
- Purpose: Map country names to ISO3 codes consistently
- Examples: `src/un_report_api/app/country_iso_map.py`, `src/un_data_pipeline/data_modules/iso2_country.py`
- Pattern: Dictionary lookup (COUNTRY_TO_ISO3); uses pycountry library as fallback for missing entries

**Classification Schemes:**
- Purpose: Standardize UN resolution subject categorization
- Examples: `src/un_data_pipeline/data_modules/un_classification.py` (nested dict of topics → tags)
- Pattern: Hierarchical taxonomy (RECOMMENDATIONS → INTERNATIONAL CIVIL SERVICE → ASSIGNMENT ALLOWANCE, etc.)

**Geographical Hierarchy:**
- Purpose: Map countries to UN regions, subregions, continents
- Examples: `src/un_data_pipeline/data_modules/un_geo_hierarchy.py`
- Pattern: Nested dict lookup for three-level hierarchy

**Report Models:**
- Purpose: Enforce schema validation for API responses
- Examples: `src/un_report_api/app/models.py` (ReportResponse, YearlyRankingsResponse, etc.)
- Pattern: Pydantic BaseModel subclasses with nested structures and field validators

**Scoring Pillars:**
- Purpose: Represent three distinct dimensions of UN voting alignment (Pillar 1, 2, 3)
- Examples: Computed in dashboard_data_pipeline.py from voting records
- Pattern: Numeric scores (0-100) aggregated yearly per country; ranks derived from scores

## Entry Points

**Web Scraper:**
- Location: `src/un_data_pipeline/scraper_pipeline.py` (function `main()`)
- Triggers: Manual execution via CLI or scheduled job
- Responsibilities: Load existing data, scrape new resolutions, tag with classification, write updated CSV, sync to Supabase

**Dashboard Pipeline:**
- Location: `src/un_data_pipeline/dashboard_data_pipeline.py` (module execution: `python -m src.un_data_pipeline.dashboard_data_pipeline`)
- Triggers: Scheduled after scraper completes; can run independently
- Responsibilities: Read Supabase, compute aggregates and scores, write CSVs to `data/processed/` and project root

**FastAPI Server:**
- Location: `src/un_report_api/app/main.py` (FastAPI instance `app`)
- Triggers: `uvicorn main:app --host 0.0.0.0 --port 8000 --reload` or Docker startup
- Responsibilities: Listen for HTTP requests, route to endpoint handlers, validate inputs, serialize responses

**Dockerfile Entry:**
- Runs FastAPI on port 8000 within container; scraper and dashboard are manual/scheduled jobs

## Error Handling

**Strategy:** Layered validation with graceful fallbacks

**Patterns:**

- **Request Validation:** Pydantic models reject invalid requests at FastAPI layer (400 Bad Request)
- **Year Bounds:** `validate_year_params()` dependency enforces MIN_YEAR_CONSTRAINT (1946) and MAX_YEAR_CONSTRAINT (current year - 1)
- **Missing Data:** FileNotFoundError caught and re-raised as HTTPException 503 (Service Unavailable)
- **Data Errors:** ValueError caught during report generation, re-raised as 404 (not found) or 400 (bad request)
- **Supabase Fallback:** If Supabase unavailable, report_generator attempts fallback to local CSVs
- **Logging:** All errors logged with context (country, year range, error type) for diagnostics
- **Unexpected Errors:** Generic 500 error returned with error type hint; full traceback logged

## Cross-Cutting Concerns

**Logging:**
- Strategy: Python standard logging with configured handlers (console + file)
- Configuration: `LOG_LEVEL` environment variable controls verbosity (INFO default)
- Files: Logs written to `logs/un_scraper_tagger.log` and `logs/` directory
- Patterns: Request logging middleware in FastAPI (logs method, path, status code)

**Validation:**
- Input validation: Pydantic models (FastAPI layer) enforce type, range, pattern constraints
- Data validation: Dashboard pipeline checks for missing columns, empty DataFrames
- Business logic: Year ranges validated against MIN/MAX constraints; country ISO codes validated against mappings

**Authentication:**
- CORS only (no authentication required)
- Restricted to single origin: https://datadrivendecisionlab.com
- Supabase credentials stored in environment variables (SUPABASE_URL, SUPABASE_KEY)
- OpenAI API key stored in environment (API_KEY for scraper classification)

**Data Consistency:**
- Pipeline idempotent via incremental deduplication (token-based)
- Rankings computed deterministically from annual_scores.csv
- Similarity scores computed once during pipeline, not recomputed per request
