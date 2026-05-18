# External Integrations

**Analysis Date:** 2026-05-18

## APIs & External Services

**Large Language Model (LLM):**
- OpenAI GPT-4o-mini / GPT-4o
  - SDK/Client: `openai` Python package v1.0.0+
  - Auth: Environment variable `API_KEY` or `OPENAI_API_KEY`
  - Usage: Tag and classify UN resolutions by subject (main tags, subtags), geographic location (continent, subregion, ISO country code)
  - Files: `src/un_data_pipeline/scraper_pipeline.py`, `src/un_report_api/app/services/veto_tagging.py`, `src/un_report_api/app/services/comprehensive_veto_regeneration.py`, `src/un_report_api/app/services/llm/runtime.py`
  - Default model: `gpt-4o-mini`
  - Temperature: 0.2-0.3 (deterministic)
  - Retry strategy: Exponential backoff with jitter on rate limit and connection errors
  - Max tokens: 1000 per call

**UN Digital Library Web Service:**
- UN Digital Library portal (https://documents.un.org)
  - Access method: Web scraping via Selenium
  - Client: Selenium WebDriver with BeautifulSoup4 HTML parsing
  - Functionality: Fetches UN voting resolutions and session meeting data
  - Files: `src/un_data_pipeline/scraper_pipeline.py`

## Data Storage

**Databases:**
- Turso (LibSQL) - Primary production database
  - Connection: Environment variables `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`
  - Client: `libsql-experimental` Python package v0.0.5+ (deferred import)
  - Fallback: HTTP API via custom `turso_http.py` client (used on Windows where libsql-experimental won't build)
  - Implementation: `src/un_data_pipeline/turso_http.py` provides TursoHTTPConnection class mimicking DB-API 2.0 cursor interface
  - Tables: `un_votes_with_sc` (configurable via `PIPELINE_SOURCE_TABLE`)
  - Auto-commit: Turso auto-commits; `commit()` is no-op in HTTP client

**File Storage:**
- Local filesystem CSV files (primary for API data loading)
  - Location: `src/un_report_api/app/required_csvs/`
  - Files used:
    - `annual_scores.csv` - Yearly voting scores by country
    - `pairwise_similarity_yearly.csv` - Country-to-country voting alignment
    - `topic_votes_yearly.csv` - Voting data aggregated by topic
    - `country_classifications_2023.csv` - Country classification metadata
    - `UN_Country_Region_Mapping_clean.csv` - Geographic region mapping
  - Loaded via: `src/un_report_api/app/services/data_loader.py` (TursoDataLoader class)

**Caching:**
- None detected

## Authentication & Identity

**Auth Provider:**
- Custom environment variable-based approach
- No OAuth/OIDC provider
- Turso authentication: Bearer token in HTTP Authorization header
- OpenAI authentication: API key passed to OpenAI client constructor

## Monitoring & Observability

**Error Tracking:**
- None detected (no Sentry, Rollbar, or similar integration)

**Logs:**
- File-based logging to `logs/un_scraper_tagger.log` (scraper pipeline)
- Console logging via Python logging module
- Log level configurable via `LOG_LEVEL` environment variable
- Format: `%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s`
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 79-88), `src/un_report_api/app/main.py` (logging setup)

## CI/CD & Deployment

**Hosting:**
- Fly.io (primary)
  - Config: `src/un_report_api/fly.toml`
  - App name: `un-report-api`
  - Primary region: CDG (Paris)
  - Force HTTPS enabled
  - Concurrency limits: 500 soft, 550 hard
  - Memory: 1GB, CPU: 1 shared

**CI Pipeline:**
- GitHub Actions (2 workflows)
  1. **Monthly Pipeline Runner** (`.github/workflows/main.yml`)
     - Trigger: 1st of each month at 02:00 UTC
     - Environment: Ubuntu latest
     - Python: 3.10
     - Runs: Full pipeline processing
  
  2. **UN Scraper Every 6 Days** (`.github/workflows/un-scraper-6days.yml`)
     - Trigger: Every 6 days at 02:00 UTC (manual dispatch available)
     - Environment: Ubuntu latest
     - Python: 3.11
     - Setup: Installs Chrome browser for Selenium automation
     - Includes DNS checks for OpenAI API availability
     - Runs: Web scraping and data collection

- Google Cloud Build (detected in `cloudbuild.yaml`, not analyzed)

## Environment Configuration

**Required env vars for scraper pipeline:**
- `TURSO_DATABASE_URL` - LibSQL connection string (format: `libsql://your-database-name.turso.io`)
- `TURSO_AUTH_TOKEN` - Turso authentication token
- `API_KEY` - OpenAI API key (also checked as `OPENAI_API_KEY`)

**Optional env vars:**
- `LOG_LEVEL` - Default: `INFO` (options: DEBUG, INFO, WARNING, ERROR)
- `PIPELINE_SOURCE_TABLE` - Default: `un_votes_with_sc`

**Secrets location:**
- GitHub Actions Secrets (for CI/CD)
  - `API_KEY` - OpenAI API key
  - `TURSO_DATABASE_URL` - Database connection URL (previously SUPABASE_URL, migrated to Turso)
  - `TURSO_AUTH_TOKEN` - Database auth token (previously SUPABASE_KEY, migrated to Turso)
- `.env` file locally (NEVER committed to version control)

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

## Legacy Integrations (Deprecated)

**Supabase (Postgres):**
- **Status:** Migrated to Turso as of recent commits
- **Previously used:** `supabase` Python package for Postgres database connection
- **References:** Lingering in `src/un_report_api/app/supabase_client.py` (kept for reference, not used)
- **API URL:** Previously used `SUPABASE_URL` environment variable
- **Auth:** Previously used `SUPABASE_KEY` environment variable
- **Reason for migration:** Turso (LibSQL/SQLite) provides simpler, more cost-effective serverless database with HTTP API support for Windows compatibility

---

*Integration audit: 2026-05-18*
