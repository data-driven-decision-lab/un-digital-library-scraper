# External Integrations

**Analysis Date:** 2026-03-18

## APIs & External Services

**Large Language Models:**
- OpenAI GPT API - LLM-based classification and analysis
  - SDK/Client: `openai>=1.0.0`
  - Model: `gpt-4o-mini`
  - Auth: `API_KEY` environment variable
  - Usage:
    - Subject matter tagging of UN resolutions (scraper_pipeline.py)
    - Geo-tagging resolutions with country/region/continent (scraper_pipeline.py)
    - Security Council veto resolution summaries (veto_tagging.py)
    - LLM runtime for structured responses (services/llm/runtime.py)
  - Retry logic: 3 retries on validation failure with exponential backoff
  - Error handling: APIConnectionError and RateLimitError caught and logged

**Web Scraping:**
- UN Digital Library (`digitallibrary.un.org`)
  - Purpose: Source of all UN voting resolution data
  - Method: Selenium + BeautifulSoup web scraping
  - Auth: None (public website)
  - Implementation: `src/un_data_pipeline/scraper_pipeline.py` (lines 54-68)

## Data Storage

**Databases:**
- Supabase PostgreSQL
  - Connection: `SUPABASE_URL` and `SUPABASE_KEY` environment variables
  - Client: `supabase>=2.0.0` Python SDK
  - Tables used:
    - `un_votes_with_sc` - Raw UN voting data with Security Council context (source table)
  - Implementation:
    - `src/un_report_api/app/supabase_client.py` - SupabaseDataLoader class
    - `src/un_data_pipeline/scraper_pipeline.py` - Data retrieval and storage
    - `src/un_data_pipeline/dashboard_data_pipeline.py` - Dashboard data pipeline
  - Data load methods:
    - `load_data_from_supabase()` with pagination (page_size=1000, max_retries=3)
    - `get_links_from_supabase()` - Fetch existing links to avoid duplicates
    - `get_all_data_from_supabase()` - Fetch all existing data for incremental updates

**File Storage:**
- Local filesystem only
  - CSV files: `src/un_report_api/app/required_csvs/` directory contains:
    - `annual_scores.csv` - Country pillar scores by year
    - `topic_votes_yearly.csv` - Voting patterns by topic and year
    - `pairwise_similarity_yearly.csv` - Country voting alignment scores
    - `country_classifications_2023.csv` - OECD, G20, GDP, population classifications
    - `UN_Country_Region_Mapping_clean.csv` - Country to UN region mapping
  - Security Council data: `src/un_report_api/app/sc_data/` directory
    - `fully_enhanced_veto_data.csv` - Veto resolutions with LLM-generated summaries
    - `un_votes_with_sc_rows.csv` - SC voting records

**Caching:**
- None detected

## Authentication & Identity

**Auth Provider:**
- Custom/None - API uses environment variables for service authentication
  - Supabase: JWT token in `SUPABASE_KEY` environment variable
  - OpenAI: API key in `API_KEY` environment variable
  - No end-user authentication on API endpoints
  - CORS enabled for: `https://datadrivendecisionlab.com` (main.py, line 68)

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, Datadog, etc.)

**Logs:**
- Python logging module with:
  - Console output via StreamHandler
  - File output to `logs/un_scraper_tagger.log`
  - Configurable level via `LOG_LEVEL` environment variable (defaults to INFO)
  - Format: `%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s`
  - HTTP request/response logging middleware in FastAPI (main.py, lines 84-89)

## CI/CD & Deployment

**Hosting:**
- Google Cloud Run (serverless container platform)
  - Region: `europe-west1`
  - Platform: managed
  - Authentication: Allow unauthenticated access

**CI Pipeline:**
- Google Cloud Build (`cloudbuild.yaml`)
  - Trigger: Likely on git push (config not shown)
  - Steps:
    1. Build Docker image: `gcr.io/$PROJECT_ID/unreportapi:$COMMIT_SHA`
    2. Push to Container Registry
    3. Deploy to Cloud Run with automatic rollout

## Environment Configuration

**Required env vars:**
- `SUPABASE_URL` - Supabase project URL (https://gjakiqtayqltssvbzasd.supabase.co)
- `SUPABASE_KEY` - Supabase service role key (has development fallback in code)
- `API_KEY` - OpenAI API key (has development fallback in code)
- `LOG_LEVEL` - Logging verbosity (optional, defaults to INFO)

**Secrets location:**
- Environment variables in `.env` file (root directory)
- `.env` is in `.gitignore` and not committed
- Development fallbacks hardcoded in `supabase_client.py` (lines 20-26) - SECURITY RISK in development

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected (unidirectional data flow from Supabase to API)

---

*Integration audit: 2026-03-18*
