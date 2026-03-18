# Technology Stack

**Analysis Date:** 2026-03-18

## Languages

**Primary:**
- Python 3.11 - All backend code, data pipeline, API, and scraping

**Secondary:**
- YAML - Docker and CI/CD configuration (cloudbuild.yaml, Dockerfile)

## Runtime

**Environment:**
- Python 3.11 (slim base image from Docker)
- uvicorn 0.24.0+ - ASGI web server

**Package Manager:**
- pip (Python)
- Lockfile: requirements.txt present (no lock file format used)

## Frameworks

**Core:**
- FastAPI 0.104.0+ - REST API framework for serving reports and analysis
- Uvicorn 0.24.0+ - ASGI server for FastAPI application

**Data Processing:**
- Pandas 1.5.0+ - Data manipulation and CSV handling
- NumPy 1.22.0+ - Numerical computations
- scikit-learn 1.1.0+ - Machine learning utilities

**Web Scraping:**
- Selenium 4.12.0+ - Browser automation for UN Digital Library scraping
- BeautifulSoup4 4.12.2+ - HTML parsing and extraction
- webdriver-manager 4.0.2+ - Automated Chrome driver management

**Testing:**
- No test framework detected in configuration

**Build/Dev:**
- Docker - Containerization for Cloud Run deployment
- Google Cloud Build - CI/CD pipeline

## Key Dependencies

**Critical:**
- openai 1.0.0+ - OpenAI API integration for LLM-based tagging and veto analysis (gpt-4o-mini model)
- supabase 2.0.0+ - Backend database client for data persistence and retrieval
- pydantic 2.0.0+ - Data validation and schema enforcement via Pydantic models

**Infrastructure:**
- python-dotenv 0.21.0+ - Environment variable loading from .env files
- requests 2.28.0+ - HTTP requests library
- tqdm 4.64.0+ - Progress bars for long-running operations
- pycountry 22.3.5+ - ISO country codes and names
- pytz 2022.7+ - Timezone handling for date operations
- dataclasses 0.6+ - Data structure support (built-in with Python 3.7+)

## Configuration

**Environment:**
- Environment variables via `.env` file (not in git)
- Required vars: `SUPABASE_URL`, `SUPABASE_KEY`, `API_KEY` (OpenAI)
- Optional: `LOG_LEVEL` (defaults to INFO)

**Build:**
- `Dockerfile` - Python 3.11-slim image with optimizations:
  - `PYTHONDONTWRITEBYTECODE=1` - No .pyc file generation
  - `PYTHONUNBUFFERED=1` - Unbuffered stdout/stderr
  - Single-stage build with cached layer for requirements
  - Expose port 8080 for API

- `cloudbuild.yaml` - Google Cloud Build pipeline:
  - Build Docker image with commit SHA tag
  - Push to Google Container Registry
  - Deploy to Cloud Run (europe-west1 region, managed, unauthenticated)

## Platform Requirements

**Development:**
- Python 3.11 interpreter
- pip package manager
- Virtual environment (venv)
- Chrome/Chromium browser (for Selenium scraping)
- Internet connection (for OpenAI API and Supabase)

**Production:**
- Google Cloud Run (serverless container platform)
- Google Container Registry (image storage)
- Supabase PostgreSQL database (external)
- OpenAI API account with valid API key

---

*Stack analysis: 2026-03-18*
