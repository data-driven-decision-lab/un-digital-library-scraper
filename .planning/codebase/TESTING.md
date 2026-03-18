# Testing Patterns

**Analysis Date:** 2026-03-18

## Test Framework

**Runner:**
- **No test framework configured** - No pytest.ini, pyproject.toml, setup.cfg, or tox.ini found
- **No test files detected** - No `*_test.py` or `*_spec.py` files found in codebase
- **Testing gap identified** - Codebase has zero automated test coverage

**Assertion Library:**
- Not configured (no testing framework present)

**Run Commands:**
```bash
# No test command available - testing infrastructure not established
# To add testing, would need:
# pytest                    # Run all tests
# pytest -v                 # Verbose output
# pytest --cov             # Coverage report
# pytest -k "test_name"    # Run specific test
```

## Test File Organization

**Location:**
- **Not applicable** - No tests exist in codebase
- Recommendation: Co-locate tests with source code or use separate `tests/` directory
- Pattern would be: `src/un_report_api/app/report_generator.py` paired with `tests/test_report_generator.py`

**Naming:**
- Not established (no test files found)
- Recommended: `test_*.py` prefix for pytest discovery: `test_report_generator.py`, `test_models.py`, etc.

**Structure:**
- Not established

## Test Structure

**Suite Organization:**
- **Not implemented** - No existing test patterns to reference

**Recommended pattern for FastAPI endpoints:**
```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

class TestReportEndpoint:
    def test_get_country_report_valid_iso(self):
        response = client.get("/report/USA?start_year=2009&end_year=2013")
        assert response.status_code == 200

    def test_get_country_report_invalid_iso(self):
        response = client.get("/report/INVALID?start_year=2009&end_year=2013")
        assert response.status_code == 400
```

**Patterns needed:**
- Setup/teardown for test fixtures
- Isolation of external dependencies (Supabase, file I/O)
- Assertion patterns for API responses

## Mocking

**Framework:**
- **Not detected** - Would use `unittest.mock` or `pytest-mock` if testing were implemented

**Patterns:**
- None currently used (no tests to establish patterns)

**What to Mock:**
- Supabase client calls (`SupabaseDataLoader`)
- File I/O operations (CSV loading)
- External API calls (OpenAI for LLM enhancement)
- Environment variables

**What NOT to Mock:**
- Pydantic model validation
- Core data transformation logic
- Business rule calculations

**Recommended mocking approach:**
```python
from unittest.mock import patch, MagicMock
import pytest

@pytest.fixture
def mock_supabase():
    with patch('supabase_client.supabase_loader') as mock:
        mock.load_annual_scores.return_value = pd.DataFrame({
            'country_code': ['USA', 'CHN'],
            'year': [2013, 2013]
        })
        yield mock

def test_generate_report_with_mock_supabase(mock_supabase):
    report = generate_report('USA', 2009, 2013)
    assert report is not None
```

## Fixtures and Factories

**Test Data:**
- **Not implemented** - No fixtures exist

**Recommendation for test data factories:**
```python
import factory
from models import ReportResponse, ReportMetadata

class ReportMetadataFactory(factory.Factory):
    class Meta:
        model = ReportMetadata

    country_iso3 = "USA"
    country_name = "United States of America"
    start_year = 2009
    end_year = 2013
```

**Location:**
- Would be: `tests/factories.py` or `tests/conftest.py` (pytest fixtures)

## Coverage

**Requirements:**
- **Not enforced** - No coverage configuration present
- **Current coverage: 0%** - No tests exist

**View Coverage:**
```bash
# Once testing is implemented:
pytest --cov=src --cov-report=html    # Generate HTML report
pytest --cov=src --cov-report=term    # Terminal report
```

## Test Types

**Unit Tests:**
- **Not implemented**
- **Scope:** Individual functions like `safe_get_value()`, `calculate_perc_change()`, `standardize_col_names()`
- **Approach:** Test with various inputs (normal, edge cases, None values, type mismatches)

**Example unit test targets:**
- `report_generator.py`: `safe_get_value()` (line 41), `calculate_perc_change()` (line 65), `scale_similarity()` (line 93)
- `ranking_generator.py`: `standardize_col_names()` (line 33), `calculate_rankings_for_year()` (line 38)
- `models.py`: Pydantic model validation

**Integration Tests:**
- **Not implemented**
- **Scope:** Full API endpoint flows with mocked Supabase/file I/O
- **Approach:** End-to-end request/response validation

**Example integration test targets:**
- `GET /report/{country_iso}` endpoint with various year ranges
- `GET /rankings/{year}` endpoint validation
- `GET /health` endpoint
- `GET /sc/veto_analysis` endpoint
- Error handling for missing data, invalid inputs

**E2E Tests:**
- **Not implemented**
- **Framework:** Would use pytest with TestClient for FastAPI testing
- **Scope:** Full system testing against live/staging Supabase instance

## Common Patterns

**Async Testing:**
- **Not implemented**
- FastAPI endpoints are async and would require:

```python
import pytest
from fastapi.testclient import TestClient

@pytest.mark.asyncio
async def test_async_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
```

**Error Testing:**
- **Not implemented**
- Pattern for testing error cases in `main.py` (lines 151-179):

```python
def test_country_report_file_not_found(mock_supabase):
    """Test 503 error when data files missing"""
    with patch('report_generator.generate_report',
               side_effect=FileNotFoundError('annual_scores.csv')):
        response = client.get("/report/USA?start_year=2009&end_year=2013")
        assert response.status_code == 503
        assert "required data file" in response.json()['detail']

def test_country_report_invalid_year(mock_supabase):
    """Test 404 error for unknown country"""
    with patch('report_generator.generate_report',
               side_effect=ValueError("Country ISO 'XYZ' not found")):
        response = client.get("/report/XYZ?start_year=2009&end_year=2013")
        assert response.status_code == 404
```

## Critical Test Gaps

**High Priority Areas (0% coverage):**

1. **Data Access Layer (`supabase_client.py`, `report_generator.py`)**
   - `SupabaseDataLoader.load_annual_scores()` - CSV loading logic
   - `SupabaseDataLoader.load_pairwise_similarity()` - Data transformation
   - `load_un_region_mapping_from_supabase()` - Region mapping logic
   - Missing tests for: file not found, empty data, malformed CSV, Supabase connection failures

2. **API Endpoints (`main.py`)**
   - `GET /report/{country_iso}` - Core business endpoint
   - `GET /rankings/{year}` - Rankings generation
   - `GET /sc/veto_analysis` - Security Council analysis
   - `GET /sc/vote_analysis` - Vote analysis
   - Missing tests for: invalid inputs, missing data, year constraints (1946-current)

3. **Report Generation (`report_generator.py`)**
   - `generate_report()` - Main report logic (line 191)
   - `safe_get_value()` - Value retrieval with fallback (line 41)
   - `calculate_perc_change()` - Percentage calculations (line 65)
   - `scale_similarity()` - Cosine similarity scaling (line 93)
   - Missing tests for: None handling, NaN/Infinity in floats, type conversions

4. **Rankings Generation (`ranking_generator.py`)**
   - `calculate_rankings_for_year()` - Ranking logic (line 38)
   - `get_rankings_for_pillar()` - Multi-pillar ranking (line 52)
   - `generate_yearly_rankings()` - Main ranking endpoint (line 126)
   - Missing tests for: missing data, tie-breaking, year-over-year changes

5. **Data Validation (`models.py`)**
   - Pydantic model field validation
   - Type coercion for Optional fields
   - Dynamic constraint validation (`MAX_YEAR_CONSTRAINT`)
   - Missing tests for: invalid field types, missing required fields, boundary values

6. **Security Council Analysis (`services/analysis_service.py`, `simple_veto_endpoint.py`)**
   - JSON data loading and parsing
   - Veto pattern analysis
   - Power dynamic classification
   - Missing tests for: malformed JSON, missing fields, empty datasets

---

*Testing analysis: 2026-03-18*

**Recommendation:** Implement testing infrastructure as high priority. Current zero test coverage creates risk for regression and makes refactoring dangerous.
