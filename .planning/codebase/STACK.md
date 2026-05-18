# Technology Stack

**Analysis Date:** 2026-05-18

## Languages

**Primary:**
- Python 3.10+ - Core pipeline, scraping, and API (used in GitHub Actions workflows, requirements.txt specifies 3.10+)

## Runtime

**Environment:**
- Python 3.10+ (verified in `.github/workflows/main.yml` and `.github/workflows/un-scraper-6days.yml`)
- Ubuntu Linux (for CI/CD via GitHub Actions)
- Windows compatible (libsql-experimental fallback to HTTP client for Windows builds)

**Package Manager:**
- pip - Manages dependencies from `requirements.txt`
- Lockfile: Not detected (using pinned versions in requirements.txt only)

## Frameworks

**Core:**
- FastAPI 0.104.0+ - REST API framework for UN Report API server
- Uvicorn 0.24.0+ (with standard extras) - ASGI web server

**Data Processing:**
- pandas 1.5.0+ - Data manipulation and CSV/DataFrame processing
- numpy 1.22.0+ - Numerical operations and array handling
- scikit-learn 1.1.0+ - Cosine similarity calculations for vote analysis

**Scraping & Browser Automation:**
- Selenium 4.12.0+ - Web browser automation for UN Digital Library scraping
- BeautifulSoup4 4.12.2+ - HTML/XML parsing
- webdriver-manager 4.0.2+ - Automatic ChromeDriver management

**Testing:**
- Not detected

**Build/Dev:**
- Not detected

## Key Dependencies

**Critical:**
- libsql-experimental 0.0.5+ - LibSQL (SQLite) client for Turso database. Deferred import in modules to handle Windows build failures gracefully. Falls back to HTTP API (`turso_http.py`) when unavailable.
- openai 1.0.0+ - OpenAI API client for GPT-based classification and tagging (used in `scraper_pipeline.py`, `veto_tagging.py`, `comprehensive_veto_regeneration.py`)
- pydantic 2.0.0+ - Data validation and structured API request/response modeling
- python-dotenv 0.21.0+ - Environment variable loading from `.env` files

**Infrastructure:**
- requests 2.28.0+ - HTTP client library (used by Selenium and general HTTP operations)
- tqdm 4.64.0+ - Progress bars for long-running operations

**Data & Utilities:**
- pytz 2022.7+ - Timezone handling for date parsing
- pycountry 22.3.5+ - ISO country code and region lookups
- dataclasses 0.6 - Structured data definition (Python 3.7+ stdlib fallback)

## Configuration

**Environment:**
- Loaded via python-dotenv from `.env` file (see `.env.example`)
- Required environment variables:
  - `TURSO_DATABASE_URL` - LibSQL connection URL for Turso database
  - `TURSO_AUTH_TOKEN` - Authentication token for Turso
  - `API_KEY` - OpenAI API key (also checked as `OPENAI_API_KEY`)
  - `LOG_LEVEL` - Logging verbosity (INFO, DEBUG, WARNING, ERROR)
  - `PIPELINE_SOURCE_TABLE` - Default: `un_votes_with_sc`

**Build:**
- Dockerfile - Container image for API deployment to Fly.io
- fly.toml - Fly.io deployment configuration (`src/un_report_api/fly.toml`)
- cloudbuild.yaml - Google Cloud Build configuration (detected, contents not analyzed)

## Platform Requirements

**Development:**
- Python 3.10+
- Chrome/Chromium browser (for Selenium automation)
- pip package manager
- Windows, macOS, or Linux compatible (Windows requires HTTP fallback for libsql-experimental)

**Production:**
- Deployment target: Fly.io (verified in `fly.toml`)
- Container platform: Docker-based deployment
- Database: Turso (LibSQL) cloud database
- API port: 8080 (configured in Fly.io)
- Region: CDG (Paris) primary region
- Concurrency: 550 hard limit, 500 soft limit connections
- Memory: 1GB
- CPU: 1 shared CPU

---

*Stack analysis: 2026-05-18*
