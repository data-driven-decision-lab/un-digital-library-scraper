# Coding Conventions

**Analysis Date:** 2026-03-18

## Naming Patterns

**Files:**
- Lowercase snake_case for Python modules: `scraper_pipeline.py`, `report_generator.py`, `supabase_client.py`
- Class files typically match class names: `UnClassificationMapper` in `un_classification_mapper.py`
- Data module files are lowercase: `un_classification.py`, `iso2_country.py`, `un_geo_hierarchy.py`

**Functions:**
- Lowercase snake_case for all functions: `generate_report()`, `calculate_perc_change()`, `load_annual_scores()`
- Async functions follow same convention: `async def log_requests()`, `async def validate_year_params()`
- Helper functions prefixed with underscore for private use: `_load_data()`, `_get_default_data_path()`
- Safe/wrapper functions often start with `safe_`: `safe_get_value()`, `safe_float()`, `safe_str()`, `safe_int()`

**Variables:**
- Lowercase snake_case: `country_iso`, `start_year`, `end_year`, `api_logger`, `project_root`
- Constants in UPPERCASE: `MIN_YEAR_CONSTRAINT`, `MAX_YEAR_CONSTRAINT`, `P5_ISO_CODES`, `DECIMAL_PLACES`
- Dictionary keys use lowercase snake_case: `country_iso`, `world_avg_pillar_1_score`, `is_oecd`

**Types:**
- Use modern Python type hints throughout: `Optional[int]`, `Dict[str, Any]`, `List[Dict]`, `Tuple[str, str]`
- Import from `typing` module: `from typing import Dict, List, Optional, Tuple, Any`
- Pydantic models for API contracts: `BaseModel` with `Field` for validation

## Code Style

**Formatting:**
- No automated formatter detected (no .prettierrc, pylint config, or black config files)
- Follows PEP 8 style guide by convention
- Lines appear to follow reasonable length limits (typically 80-120 characters)
- Consistent indentation with 4 spaces

**Linting:**
- No linting configuration found (no .pylintrc, .flake8, or similar)
- However, code follows PEP 8 conventions
- Type hints are consistently used throughout

## Import Organization

**Order:**
1. Standard library imports: `import os`, `import json`, `import logging`, `import re`, `from datetime import datetime`
2. Third-party library imports: `import pandas as pd`, `import numpy as np`, `from fastapi import FastAPI`, `from pydantic import BaseModel`
3. Local imports: `from report_generator import generate_report`, `from models import ReportResponse`, `from services.analysis_service import SecurityCouncilAnalysisService`

**Pattern Examples:**
- FastAPI imports: `from fastapi import FastAPI, Path, Query, HTTPException, Depends, Request`
- Data science stack: `import pandas as pd`, `import numpy as np`, `from sklearn.metrics.pairwise import cosine_similarity`
- Type imports: `from typing import Dict, List, Optional, Tuple, Any`
- Local relative imports: `from report_generator import` (same directory) or `from services.analysis_service import` (subdirectory)

**Path Aliases:**
- None detected. Relative imports used for package structure navigation.

## Error Handling

**Patterns:**
- Try-except blocks with specific exception handling: `except FileNotFoundError as e:`, `except ValueError as e:`, `except KeyError as e:`
- Broad exception fallbacks: `except Exception as e:` for unexpected errors
- Multiple exception types in single handler: `except (ValueError, TypeError):`
- HTTPException raised in FastAPI endpoints: `raise HTTPException(status_code=400, detail=str(e))`
- ValueError raised in data processing functions: `raise ValueError(f"Error message: {e}")`

**File examples:**
- `report_generator.py`: Lines 43-63 show nested try-except with specific exception handling for data retrieval
- `main.py`: Lines 160-179 show FastAPI error handling pattern with specific status codes (404, 503, 500)
- `supabase_client.py`: Lines 53-55 show catch-and-re-raise pattern with logging

**Logging in error handlers:**
```python
except FileNotFoundError as e:
    api_logger.error(f"Prerequisite data file not found: {e.filename}", exc_info=True)
    raise HTTPException(status_code=503, detail=...)
```

## Logging

**Framework:** Standard Python `logging` module

**Patterns:**
- Module-level logger initialization: `logger = logging.getLogger(__name__)` (see `data_loader.py` line 11, `services/analysis_service.py` line 30)
- API-level logger: `api_logger = logging.getLogger("un_report_api")` (see `main.py` lines 75-81)
- Log levels used appropriately:
  - `logger.info()` for standard operations: "Successfully loaded...", "Processing..."
  - `logger.warning()` for recoverable issues: "Column not found", "Data not available"
  - `logger.error()` for failures with exc_info: `api_logger.error(f"...", exc_info=True)`
  - `logger.debug()` for detailed troubleshooting: `logger.debug(f"Processing rankings for pillar: {pillar_name}")`

**Logging configuration:**
- FastAPI app configures handlers at startup (`main.py` lines 74-81)
- Format: `'%(asctime)s - %(name)s - %(levelname)s - %(message)s'` or `'%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'`
- File logging to `logs/un_scraper_tagger.log` in pipeline modules
- Environment-based level control: `LOG_LEVEL` env var (`scraper_pipeline.py` line 72)

## Comments

**When to Comment:**
- File-level docstrings describing module purpose (see all files: `"""FastAPI application..."""`)
- Function docstrings for public functions with parameters and returns (see `data_loader.py` lines 18-23, `ranking_generator.py` lines 38-42)
- Inline comments explaining non-obvious logic: `# Rank, 'min' method assigns the same rank to ties...`
- Section comments marking major code blocks: `# --- CORS Middleware Configuration ---`, `# --- Dependency for Year Validation ---`

**When NOT to comment:**
- Obvious code (variable assignments, simple operations)
- Code that should be self-documenting through clear naming

**JSDoc/TSDoc:**
- Not used (Python codebase, no TypeScript)
- Docstrings follow Python conventions with triple quotes

## Function Design

**Size:** Typically 30-150 lines
- Shorter for utility functions: `safe_get_value()` ~15 lines, `calculate_perc_change()` ~12 lines
- Moderate for business logic: `generate_report()` ~150+ lines, `get_yearly_rankings_api()` ~40 lines
- Larger for data processing pipelines: `scraper_pipeline.py` functions span 50-200+ lines

**Parameters:**
- Usually 1-5 parameters per function
- Use type hints for all parameters: `def generate_report(country_iso: str, start_year: int, end_year: int) -> dict:`
- Optional parameters have defaults: `def load_annual_scores(self, year: Optional[int] = None) -> pd.DataFrame:`
- FastAPI endpoints use dependency injection: `year_params: Dict[str, int] = Depends(validate_year_params)`

**Return Values:**
- Explicit return types specified: `-> dict`, `-> pd.DataFrame`, `-> Optional[str]`, `-> Dict[str, Any]`
- Tuple returns for multiple values: `-> Tuple[Dict, Optional[str]]`, `-> tuple[str, str]`
- None returned for missing data: `return None` for safe accessors
- Dictionary returns for structured data: `return {"status": "ok"}`

## Module Design

**Exports:**
- Explicit imports used throughout: `from models import ReportResponse, MIN_YEAR_CONSTRAINT, ...`
- Module-level constants and classes at top of file
- Single responsibility per module (e.g., `report_generator.py` for report logic, `models.py` for Pydantic models, `supabase_client.py` for data access)

**Barrel Files:**
- `__init__.py` files exist but content not shown (see `app/__init__.py`, `services/__init__.py`)
- Likely empty or minimal (common pattern for package initialization)

**Class Organization:**
- Pydantic models define API contracts: `class ReportResponse(BaseModel):`
- Service classes encapsulate domain logic: `class SupabaseDataLoader:`, `class SecurityCouncilAnalysisService:`
- Methods organized as: `__init__`, private methods (`_method()`), then public methods

---

*Convention analysis: 2026-03-18*
