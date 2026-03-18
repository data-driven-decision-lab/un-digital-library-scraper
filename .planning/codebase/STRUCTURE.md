# Codebase Structure

**Analysis Date:** 2026-03-18

## Directory Layout

```
un-digital-library-scraper/
├── src/                                    # Main source code
│   ├── un_data_pipeline/                   # Data scraping & aggregation pipeline
│   │   ├── scraper_pipeline.py             # Selenium scraper + LLM tagging
│   │   ├── dashboard_data_pipeline.py      # Score calculation & CSV generation
│   │   ├── data_modules/                   # Reference data (classifications)
│   │   │   ├── un_classification.py        # Subject matter tags hierarchy
│   │   │   ├── un_geo_hierarchy.py         # Geographic regions/countries
│   │   │   └── iso2_country.py             # ISO country code mappings
│   │   └── README.md
│   │
│   └── un_report_api/                      # FastAPI application
│       └── app/                            # Main API module
│           ├── main.py                     # FastAPI app, endpoint routing
│           ├── models.py                   # Pydantic response models
│           ├── report_generator.py         # Country report synthesis
│           ├── ranking_generator.py        # Annual pillar rankings
│           ├── country_iso_map.py          # Country name → ISO3 lookup
│           ├── supabase_client.py          # Supabase data loader
│           ├── simple_veto_endpoint.py     # Security Council veto endpoint
│           ├── required_csvs/              # Essential data files
│           │   ├── annual_scores.csv       # Pillar scores per country/year
│           │   └── pairwise_similarity_yearly.csv  # Country similarity matrix
│           ├── sc_data/                    # Security Council analysis data
│           │   ├── enhanced_veto_descriptions.json
│           │   ├── final_analysis_data.csv
│           │   ├── researcher_topic_report.json
│           │   └── policy_report_data.json
│           └── services/                   # API service modules
│               ├── data_loader.py          # Data file path resolution
│               ├── analysis_service.py     # Veto pattern analysis
│               ├── vote_analysis_service.py # Voting pattern analysis
│               ├── veto_tagging.py         # Veto record processing
│               ├── simple_veto_enhancement.py  # Veto data enrichment
│               ├── comprehensive_veto_regeneration.py  # Veto data regeneration
│               └── llm/                    # LLM integration for SC analysis
│                   ├── runtime.py
│                   └── schemas.py
│
├── data/                                   # Data storage
│   ├── processed/                          # Pipeline output (local staging)
│   │   └── [aggregated CSVs during pipeline]
│   └── reference/                          # Reference data
│       └── UN_Country_Region_Mapping.csv
│
├── logs/                                   # Application logs
│   └── [log files from scraper/pipeline]
│
├── .github/                                # GitHub workflows
├── requirements.txt                        # Python dependencies
├── README.md                               # Project documentation
└── Dockerfile                              # Container configuration
```

## Directory Purposes

**`src/un_data_pipeline/`:**
- Purpose: Batch data collection, enrichment, and aggregation
- Contains: Web scraper, classification pipeline, aggregation logic
- Key files: `scraper_pipeline.py` (web extraction), `dashboard_data_pipeline.py` (scoring)

**`src/un_report_api/app/`:**
- Purpose: REST API serving analytics and reports
- Contains: FastAPI application, endpoint handlers, response models
- Key files: `main.py` (routing), `report_generator.py` (synthesis), `models.py` (validation)

**`src/un_data_pipeline/data_modules/`:**
- Purpose: Reference data for classification and mapping
- Contains: Hierarchical dictionaries for UN topics, geography, country codes
- Key files: `un_classification.py`, `un_geo_hierarchy.py`, `iso2_country.py`

**`src/un_report_api/app/services/`:**
- Purpose: Specialized analysis modules
- Contains: Security Council veto/vote analysis, data loading utilities
- Key files: `analysis_service.py`, `vote_analysis_service.py`, `data_loader.py`

**`src/un_report_api/app/required_csvs/`:**
- Purpose: Essential data files for API operation
- Contains: Pre-computed annual scores and similarity matrices
- Key files: `annual_scores.csv`, `pairwise_similarity_yearly.csv`

**`src/un_report_api/app/sc_data/`:**
- Purpose: Security Council-specific analytical data
- Contains: Veto patterns, voting analysis, policy reports
- Key files: JSON files with enhanced descriptions and research reports

**`data/`:**
- Purpose: Project-wide data storage
- Contains: Processed outputs from pipeline, reference mappings
- Key files: `UN_Country_Region_Mapping.csv` (geo reference)

**`logs/`:**
- Purpose: Runtime log files
- Contains: Scraper execution logs, API request logs
- Key files: `un_scraper_tagger.log`, request/error traces

## Key File Locations

**Entry Points:**
- `src/un_data_pipeline/scraper_pipeline.py`: Main scraper entry (`main()` function)
- `src/un_data_pipeline/dashboard_data_pipeline.py`: Pipeline execution (direct module run)
- `src/un_report_api/app/main.py`: FastAPI application instance

**Configuration:**
- `requirements.txt`: Python package dependencies
- `.env` (not shown, add to root): Environment variables (API_KEY, SUPABASE_URL, SUPABASE_KEY)
- `Dockerfile`: Container configuration with FastAPI startup

**Core Logic:**
- `src/un_report_api/app/report_generator.py`: Multi-section report assembly
- `src/un_report_api/app/ranking_generator.py`: Annual rankings computation
- `src/un_data_pipeline/scraper_pipeline.py`: Web scraping + tagging (>100KB file)

**Testing:**
- No dedicated test directory; tests not present in codebase

**Data/Models:**
- `src/un_report_api/app/models.py`: Pydantic schemas for all API responses
- `src/un_data_pipeline/data_modules/`: Reference dictionaries for classification/mapping

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `scraper_pipeline.py`, `report_generator.py`)
- CSV outputs: `UPPERCASE_WITH_UNDERSCORES_YYYY-MM-DD.csv` (e.g., `UN_VOTING_DATA_RAW_WITH_TAGS_2025-03-18.csv`)
- Data files: descriptive, lowercase (e.g., `annual_scores.csv`, `pairwise_similarity_yearly.csv`)

**Directories:**
- Package dirs: `snake_case` (e.g., `un_data_pipeline`, `un_report_api`)
- Feature dirs: `descriptive_lowercase` (e.g., `data_modules`, `required_csvs`, `sc_data`)

**Functions:**
- All functions: `snake_case` (e.g., `generate_report()`, `calculate_rankings_for_year()`)
- Class methods: `snake_case` (e.g., `load_annual_scores()`, `get_enhanced_veto_descriptions_path()`)

**Variables:**
- Local variables, parameters: `snake_case` (e.g., `country_iso`, `start_year`, `df_current_year`)
- Constants: `UPPERCASE` (e.g., `MIN_YEAR_CONSTRAINT`, `MAX_YEAR_CONSTRAINT`, `P5_ISO_CODES`)
- DataFrame columns (from CSV): mixed case preserved from source (e.g., `TOTAL VOTES`, `YES COUNT`) but normalized to `snake_case` during processing

**Types:**
- Pydantic models: `PascalCase` (e.g., `ReportResponse`, `ReportMetadata`, `YearlyRankingsResponse`)
- Type hints: lowercase (e.g., `pd.DataFrame`, `Optional[str]`, `List[Dict]`)

## Where to Add New Code

**New Feature (General Approach):**
- Backend business logic: `src/un_report_api/app/` (if API-related) or `src/un_data_pipeline/` (if pipeline-related)
- New endpoint: Add route to `src/un_report_api/app/main.py`, response model to `models.py`
- New service: Add file to `src/un_report_api/app/services/` following pattern of `analysis_service.py`

**New Component/Module:**
- API component: Create file in `src/un_report_api/app/`, import in `main.py`
- Pipeline component: Create file in `src/un_data_pipeline/`, call from `scraper_pipeline.py` or `dashboard_data_pipeline.py`
- Reference data: Add to `src/un_data_pipeline/data_modules/` (dictionary/CSV)

**Utilities:**
- Shared helpers: Place in service modules (e.g., `src/un_report_api/app/services/`) or create `utils.py` in relevant package
- Data mapping functions: Add to `src/un_data_pipeline/data_modules/` for pipeline, or `src/un_report_api/app/` for API-specific

**Tests:**
- Currently no test structure; if adding tests, create `tests/` directory at project root with mirrored structure
- Pattern would follow: `tests/test_report_generator.py`, `tests/test_scraper_pipeline.py`, etc.

## Special Directories

**`src/un_report_apiold/`:**
- Purpose: Legacy API version (deprecated)
- Generated: No (committed for reference)
- Committed: Yes (for historical tracking)
- Note: Not used in current deployment; present for fallback/comparison

**`src/un_data_pipeline/__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes (auto-generated during execution)
- Committed: No (excluded via .gitignore)

**`.planning/codebase/`:**
- Purpose: GSD planning documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Generated: No (manually created)
- Committed: Yes (referenced by other GSD commands)

**`logs/`:**
- Purpose: Runtime execution logs
- Generated: Yes (created during scraper/pipeline execution)
- Committed: No (in .gitignore)

**`.github/`:**
- Purpose: GitHub workflows for CI/CD
- Generated: No (manually configured)
- Committed: Yes

---

## Implementation Patterns

**Adding a New API Endpoint:**

1. Define response model in `src/un_report_api/app/models.py`:
   ```python
   class MyNewResponse(BaseModel):
       field1: str
       field2: Optional[int] = None
   ```

2. Create route in `src/un_report_api/app/main.py`:
   ```python
   @app.get("/my_endpoint", response_model=MyNewResponse)
   async def get_my_endpoint(param: str = Query(...)):
       # Implementation
       return MyNewResponse(field1="...", field2=123)
   ```

3. If logic is complex, extract to service in `src/un_report_api/app/services/`:
   ```python
   # my_service.py
   class MyService:
       def analyze(self, param: str) -> dict:
           # Implementation
           return {"result": ...}
   ```

4. Import and use in endpoint:
   ```python
   from services.my_service import MyService

   service = MyService()
   result = service.analyze(param)
   ```

**Adding Data to Pipeline:**

1. If new CSV source, place in `src/un_report_api/app/required_csvs/`
2. Load via `SupabaseDataLoader` or direct pandas read in relevant generator module
3. Update `data_loader.py` if paths need resolution
4. Document in README.md pipeline section
