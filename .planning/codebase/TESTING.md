# Testing Patterns

**Analysis Date:** 2026-05-18

## Test Framework

**Current State:** No testing framework configured

- **Test Runner:** Not detected
- **Assertion Library:** Not detected
- **Config Files:** No `pytest.ini`, `pyproject.toml`, `setup.py`, `conftest.py`, or `tox.ini` present
- **Test Directory:** No dedicated `tests/` directory

**Recommendation:** The project currently has zero automated tests. See [Testing Setup Required](#testing-setup-required) section below.

## Test File Organization

**Current Structure:**
- No test files found in codebase
- No co-located test files (e.g., `module_test.py` alongside `module.py`)
- No separate test directory

**What Should Be:**
- Tests co-located with source or in top-level `tests/` directory
- Pattern: `tests/test_scraper_pipeline.py` or `src/un_data_pipeline/test_scraper.py`
- Naming convention: `test_*.py` or `*_test.py`

## Manual Testing Patterns Observed

**Scraper Testing:**
- `scraper_pipeline.py` is designed to be executed directly: `if __name__ == "__main__": main()`
- Includes logging at multiple levels to trace execution:
  - INFO: Major phase boundaries and progress
  - DEBUG: Detailed operation traces
  - ERROR: Failures with context
- Checkpoint function `checkpoint_progress()` writes intermediate state to allow recovery
- Session-based approach: Browser sessions reset after `SESSION_RESET_THRESHOLD = 150` requests

**API Testing:**
- `main.py` FastAPI app can be tested with `uvicorn` server
- Request logging middleware (`log_requests`) captures all HTTP operations
- Response status codes logged after completion
- No dedicated test client library integrated

**Data Validation:**
- Pydantic models auto-validate at API layer (FastAPI dependency)
- Example: `ReportResponse` validates structure before returning
- Year constraints validated via `Query()` parameters: `ge=MIN_YEAR_CONSTRAINT, le=MAX_YEAR_CONSTRAINT`

## Error Scenarios (Tested Implicitly)

**Handled but not formally tested:**

1. **API Connection Failures** (`scraper_pipeline.py:306-339`):
   ```python
   def execute_api_call(api_call_fn, max_retries=5):
       """Retry logic with exponential backoff for transient failures."""
       for attempt in range(max_retries):
           try:
               return api_call_fn(client)
           except (RateLimitError, APIConnectionError) as e:
               logger.warning(f"Attempt {attempt + 1} failed: {e}")
               time.sleep(2 ** attempt)  # Exponential backoff
           except Exception as e:
               logger.error(f"Non-retryable error: {e}")
               raise e
       raise Exception("OpenAI API request failed after max retries")
   ```
   - Catches `RateLimitError`, `APIConnectionError`
   - Retries up to 5 times with exponential backoff
   - Logs each attempt and final failure

2. **Scraper Recovery** (`scraper_pipeline.py:2248-2257`):
   ```python
   except (ConnectionResetError, ConnectionRefusedError, KeyboardInterrupt) as e:
       logger.error(f"Year {year}: Connection error during link collection: {e}")
       driver.quit()
       driver = get_driver()
       driver.get(BASE_SEARCH_URL)
   ```
   - Catches connection failures and keyboard interrupt
   - Restarts browser session
   - Continues execution

3. **Database Operations** (`scraper_pipeline.py:175-184`):
   ```python
   try:
       conn = get_turso_connection()
       conn.execute(...)
       conn.commit()
   except Exception as e:
       logger.error(f"Failed to create scraper log entry: {e}")
   ```
   - Generic exception handling with logging
   - Operations continue even if DB operations fail

## Mocking

**Current State:** No mocking framework configured

- `unittest.mock` (stdlib) not imported anywhere
- `pytest-mock` not in `requirements.txt`
- No test fixtures or factory pattern implementation

**What Should Be Mocked:**
- Selenium WebDriver: Browser interactions to avoid actual scraping
- API calls: OpenAI/Gemini API to avoid token consumption and rate limits
- Database: Turso/libsql connections to isolate data layer
- File I/O: CSV reads/writes to test data pipelines without disk access
- Network: HTTP requests to external UN Digital Library website

## Fixtures and Factories

**Current State:** No test fixtures present

**Data Patterns in Production Code:**

Example from `report_generator.py` showing data structure expectations:
```python
def safe_get_value(df, year, country_col, target_country, value_col):
    """Safely get a single value from a DataFrame for a specific year and country."""
    try:
        value = df.loc[(df['year'] == year) & (df[country_col] == target_country), value_col].iloc[0]
```

Example from `models.py` showing expected response structure:
```python
class ReportMetadata(BaseModel):
    country_iso3: str = Field(..., example="USA")
    country_name: str = Field(..., example="United States of America")
    start_year: int = Field(..., example=2009)
    end_year: int = Field(..., example=2013)
```

**What Should Exist:**
- Fixture for sample vote data (CSV with years, countries, vote counts)
- Fixture for Turso database connection mock
- Factory for generating test `ReportResponse` objects with realistic data
- Factory for creating sample Pydantic model instances

## Coverage

**Requirements:** No coverage tooling present or enforced

**Tools That Should Be Used:**
- `pytest-cov` for coverage reports
- Coverage target: Minimum 70% for core business logic (scraper, report generation)
- Core files needing coverage:
  - `scraper_pipeline.py`: 2362 lines - scrapers, LLM tagging, Turso interaction
  - `report_generator.py`: 670 lines - report generation logic
  - `analysis_service.py`: 151 lines - Security Council analysis
  - `turso_client.py`: 215 lines - database interactions

**View Coverage (when implemented):**
```bash
pytest --cov=src --cov-report=html
pytest --cov=src --cov-report=term-missing
```

## Test Types

**Unit Tests (Needed):**
- **Scope:** Individual functions in isolation
- **Examples:**
  - `test_safe_get_value()` - verify null handling in dataframe access
  - `test_safe_float()` - verify NaN/Inf conversion
  - `test_standardize_country_columns()` - verify ISO3 normalization
  - `test_tag_resolution()` - verify tag extraction from title
  - `test_scale_similarity()` - verify cosine similarity scaling (0-100)
  - `test_calculate_perc_change()` - verify percentage calculation edge cases
- **Approach:** Mock dependencies, test logic in isolation

**Integration Tests (Needed):**
- **Scope:** Multiple components working together
- **Examples:**
  - Load sample CSV → Process tags → Validate output structure
  - Query Turso → Load vote data → Generate report
  - API endpoint → Report generation → Response validation
- **Approach:** Use test fixtures, mock database, real data processing

**E2E Tests (Not Implemented):**
- **Framework:** Not in use
- **Justification:** Scraper inherently requires browser automation; data pipeline requires external APIs
- **Alternative:** Manual testing and observability through logging

## Common Patterns to Test

**Async Testing (Not Required):**
- Python `asyncio` not used in data pipeline
- FastAPI endpoints use `async` but can be tested with `TestClient` from `fastapi.testclient`

**Error Testing (Examples from Code):**

```python
# Test that year validation rejects invalid years
def test_year_validation_bounds():
    response = client.get("/report/USA?start_year=1900&end_year=2025")
    assert response.status_code == 422  # Pydantic validation error

# Test that country not found returns appropriate response
def test_country_not_found():
    response = client.get("/report/INVALID?start_year=2010&end_year=2015")
    assert response.status_code == 400  # HTTPException raised

# Test NaN handling in safe operations
def test_safe_get_value_with_nan():
    df = pd.DataFrame({"year": [2010], "country": ["USA"], "score": [np.nan]})
    result = safe_get_value(df, 2010, "country", "USA", "score")
    assert result is None
```

## Testing Setup Required

**Phase 1: Framework Setup**
```bash
# Add to requirements.txt
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-asyncio>=0.21.0  # For async endpoint testing
pytest-mock>=3.10.0
responses>=0.22.0  # For mocking HTTP requests
```

**Phase 2: Configuration**
Create `pyproject.toml` section:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "--cov=src --cov-report=term-missing:skip-covered --cov-fail-under=70"
```

**Phase 3: Test Structure**
```
tests/
├── conftest.py                    # Shared fixtures
├── test_scraper_pipeline.py       # Scraper tests
├── test_report_generator.py       # Report generation tests
├── test_api_main.py               # FastAPI endpoint tests
├── test_analysis_service.py       # Security Council analysis tests
├── test_data_loaders.py           # Turso/CSV loading tests
├── fixtures/
│   ├── sample_votes.csv           # Test voting data
│   ├── sample_response.json       # Expected API response
│   └── mock_turso_data.json       # Database fixture
└── integration/
    └── test_end_to_end_report.py  # Full pipeline tests
```

**Phase 4: Critical Test Cases**

Unit tests for `scraper_pipeline.py`:
- `test_tag_new_rows_with_valid_titles()`
- `test_get_llm_location_tags_handles_api_errors()`
- `test_execute_api_call_retries_on_rate_limit()`
- `test_standardize_country_columns_converts_to_iso3()`

Integration tests:
- `test_scraper_pipeline_upload_to_turso()` (with mock DB)
- `test_report_generation_with_sample_data()`
- `test_api_country_report_endpoint()` (with TestClient)

---

*Testing analysis: 2026-05-18*
