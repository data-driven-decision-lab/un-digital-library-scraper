# Coding Conventions

**Analysis Date:** 2026-05-18

## Naming Patterns

**Files:**
- Module files use lowercase with underscores: `scraper_pipeline.py`, `turso_client.py`, `dashboard_data_pipeline.py`
- Data module files follow same convention: `un_classification.py`, `iso2_country.py`
- API endpoints and services use lowercase: `analysis_service.py`, `report_generator.py`

**Functions:**
- Functions use lowercase with underscores (snake_case): `get_turso_connection()`, `start_scraper_log()`, `tag_new_rows()`
- Helper functions prefixed with intent: `safe_get_value()`, `safe_float()`, `extract_vote_data_from_html()`
- Private helper functions follow same snake_case convention

**Variables:**
- Local variables and parameters use snake_case: `existing_links`, `new_records_processed`, `session_request_count`
- Global constants use UPPERCASE with underscores: `DEFAULT_MODEL`, `MAX_WORKERS`, `BASE_SEARCH_URL`, `FIXED_COLUMNS`
- Dictionary keys often use lowercase snake_case: `country_stats`, `topic_stats`, `power_dynamics`

**Types and Classes:**
- Pydantic models use PascalCase: `LocationClassifications`, `ResolutionTarget`, `ReportResponse`, `SecurityCouncilAnalysisService`
- Exception classes use PascalCase: `DuplicateLinkFound`
- Type hints use standard Python conventions with `Optional`, `List`, `Dict`, `Tuple`, `Any`

## Code Style

**Formatting:**
- No formatter is explicitly configured (no `.prettierrc` or `ruff.toml`)
- Code follows PEP 8 conventions implicitly
- String formatting: F-strings are standard throughout codebase
- Line length: Generally follows implicit 88-100 character guidelines based on observed code

**Linting:**
- No explicit linting configuration files present (no `.eslintrc`, `.pylintrc`)
- Imports are organized but not strictly enforced:
  - Standard library imports at top (sys, os, logging, json)
  - Third-party packages (pandas, numpy, selenium, pydantic)
  - Relative imports for local modules (e.g., `from .data_modules.un_classification import un_classification`)

## Import Organization

**Order:**
1. Standard library imports: `sys`, `os`, `logging`, `json`, `time`, `uuid`, `re`, `csv`, `math`, `platform`
2. Data/compute libraries: `pandas as pd`, `numpy as np`
3. Web/HTTP libraries: `selenium`, `requests`, `BeautifulSoup`, `fastapi`, `uvicorn`, `openai`
4. Data validation: `pydantic` (`BaseModel`, `Field`, `validator`)
5. Utilities: `dotenv`, `tqdm`, `webdriver_manager`, `pycountry`
6. Database: `libsql_experimental` (with fallback to HTTP client on Windows)
7. Relative imports: `from .data_modules...`, `from .services...`

**Path Aliases:**
- Project-relative paths constructed with `os.path.join()` and environment-relative paths
- Environment-relative paths: `TURSO_DATABASE_URL`, `TURSO_AUTH_TOKEN` read via `os.getenv()`
- Local relative imports use dot notation: `from .data_modules.un_classification import un_classification`

## Error Handling

**Patterns:**
- Generic exception catching with logging: `except Exception as e:` followed by `logger.error(f"...")`
- Specific exceptions caught for retryable operations: `except (RateLimitError, APIConnectionError) as e:`
- Selenium-specific exceptions caught separately: `TimeoutException`, `NoSuchElementException`, `ElementNotInteractableException`, `StaleElementReferenceException`
- Custom exceptions defined with descriptive names: `DuplicateLinkFound(Exception)` with `new_links` attribute
- Network/connection errors distinguished: `(ConnectionResetError, ConnectionRefusedError, KeyboardInterrupt)`
- Environment validation at startup: `raise ValueError("GEMINI_API_KEY not found in environment variables.")`

**Error Recovery:**
- Graceful degradation: Functions return empty structures or defaults on error
- Logging of errors at appropriate levels: ERROR for critical failures, WARNING for recoverable issues, DEBUG for detailed traces
- Retry logic implemented with `execute_api_call()` wrapper and exponential backoff for API calls

## Logging

**Framework:** Python `logging` module (stdlib)

**Configuration (from `scraper_pipeline.py`):**
- Configured at module level with stream and file handlers
- File handler writes to `logs/un_scraper_tagger.log`
- Log level controlled via `LOG_LEVEL` environment variable (defaults to INFO)
- Format: `"%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"`
- Time format: `"%H:%M:%S"`

**Patterns:**
- Initial info log on startup: `logger.info("Starting Turso-native UN voting data scraper...")`
- Progress logging at major phase boundaries: `logger.info(f"\n{'='*60}\nProcessing year {year}...")`
- Debug logging for detailed operation traces: `logger.debug(f"Year {year}: Starting link collection")`
- Error logging with full exception context: `logger.error(f"Error message: {e}")`
- Warning logging for non-critical issues: `logger.warning(f"No active scraper run to update")`
- Numeric debug level available: `logger.setLevel(logging.DEBUG)`
- Dynamic level helpers: `enable_debug_logging()`, `enable_verbose_scraping()`

**In FastAPI (from `main.py`):**
- Logger obtained: `api_logger = logging.getLogger("un_report_api")`
- Request logging middleware logs all HTTP requests with method, path, query params
- Response status codes logged after completion
- Handlers check `if not api_logger.hasHandlers()` to prevent duplicate logging

## Comments

**When to Comment:**
- Docstrings required for all functions with multi-line format
- Example from `get_turso_connection()`: Full docstring with Args, Returns, Raises sections
- Complex logic steps documented inline (e.g., regex pattern compilation with explanation)
- Configuration constants documented with purpose: `MAX_CONSECUTIVE_EMPTY_PAGES = 3  # Stop after this many...`
- Section headers as visual dividers: `# ---------- Geo-Tagging Functions ----------`

**JSDoc/TSDoc:**
- Not applicable (Python codebase, not JavaScript/TypeScript)
- Pydantic model fields use `Field(description="...")` for inline documentation
- OpenAPI schema docs from Pydantic models: `Field(..., example="USA", description="3-letter ISO code")`

## Function Design

**Size:**
- Most functions 20-100 lines (moderate scope)
- Larger orchestrators like `main()` and `scraper_pipeline.py:process_and_upload_data()` exceed 100 lines (acceptable for complex workflows)
- Helper utilities are typically 5-30 lines

**Parameters:**
- Explicit parameter names with type hints: `def call_llm_api(title: str, geo_hierarchy: dict, model: str = DEFAULT_MODEL) -> ResolutionTarget:`
- Default parameters use module constants: `model: str = DEFAULT_MODEL`
- Variadic parameters avoid overuse; used when necessary: `max_retries=5`, `batch_size=30`
- Boolean flags used for optional behavior: `recent_year_only: bool = Query(False, ...)`

**Return Values:**
- Functions declare return types: `-> ResolutionTarget`, `-> List[List]`, `-> Dict[str, Any]`
- Multiple returns indicated with `Tuple`: `-> Tuple[Dict, Optional[str]]`
- None returns explicitly typed: `-> Optional[str]`
- Dataframe returns for data operations: `-> pd.DataFrame`

## Module Design

**Exports:**
- Modules designed for direct import: `from .data_modules.un_classification import un_classification`
- Data modules export dictionaries/constants: `iso2_country_code`, `geo_hierarchy`, `un_classification`
- Service modules export classes: `SecurityCouncilAnalysisService`
- Pipeline modules designed as executable scripts with `if __name__ == "__main__":` guard
- API modules export FastAPI `app` instance and endpoint functions

**Barrel Files:**
- Service `__init__.py` files import and expose key classes from submodules
- Example: `src/un_report_api/app/services/__init__.py` exposes service classes
- Allows clean imports: `from services import SecurityCouncilAnalysisService`
- Not used for data_modules (each module imported directly by name)

## API Response Patterns

**FastAPI Endpoints:**
- All endpoints decorated with HTTP method and path: `@app.get("/")`, `@app.get("/report/{country_iso3}")`
- Path parameters use FastAPI `Path()`: `country_iso3: str = Path(...)`
- Query parameters use `Query()`: `start_year: int = Query(...)`
- Response models declared: `-> ReportResponse`
- Error responses use `HTTPException`: `raise HTTPException(status_code=400, detail="Invalid year")`
- Async functions standard: `async def get_country_report_api(...)`
- Logging middleware wraps all requests: `@app.middleware("http")`

**Pydantic Models:**
- All response models inherit `BaseModel`
- Fields use `Field()` with examples and descriptions for OpenAPI
- Optional fields explicitly typed: `Optional[float] = Field(default=None, ...)`
- Nested models for complex structures: `ReportResponse` contains `ReportMetadata`, `VotingBehaviorOverall`, etc.
- Validators available but not heavily used: `@validator` decorator syntax available

## Database Patterns

**Connection Handling:**
- `get_turso_connection()` creates fresh connection per call
- Lazy imports: `libsql_experimental` imported inside function to avoid startup failures
- Fallback to HTTP client on Windows: `try libsql import / except ImportError _USE_HTTP_CLIENT = True`
- Connection obtained before each operation, not pooled
- Explicit `conn.commit()` after mutations

**Data Loading:**
- CSV files used as primary data source in API: `pd.read_csv(csv_path)`
- Turso used for logging and analytics in scraper
- Data loader pattern: `TursoDataLoader` class with methods like `load_annual_scores(year)`

---

*Convention analysis: 2026-05-18*
