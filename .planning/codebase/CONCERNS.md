# Codebase Concerns

**Analysis Date:** 2026-03-18

## Tech Debt

**Bare Exception Handlers:**
- Issue: Multiple catch-all `except:` blocks without exception type specification, silently swallowing errors and making debugging difficult
- Files:
  - `src/un_data_pipeline/scraper_pipeline.py` (lines 1868, 2286)
  - `src/un_data_pipeline/dashboard_data_pipeline.py` (lines 349, 537)
  - `src/un_report_api/app/simple_veto_endpoint.py` (line 23)
  - `src/un_report_api/app/services/comprehensive_veto_regeneration.py` (line 105)
  - `src/un_report_api/app/services/simple_veto_enhancement.py` (line 91)
  - `src/un_data_pipeline/dashboard_data_pipeline.py` (line 537)
- Impact: Errors are masked, making production debugging nearly impossible; failures propagate silently; exception-specific handling is impossible
- Fix approach: Replace all `except:` with specific exception types (e.g., `except Exception as e:`, `except TimeoutException:`, `except ValueError:`) and log the exception before handling

**Hardcoded Credentials in Development Fallback:**
- Issue: Supabase URL and JWT token are hardcoded as fallback values in development mode
- Files: `src/un_report_api/app/supabase_client.py` (lines 20-26)
- Impact: Exposes credentials in source code; reduces security posture even though commented as development fallback; JWT token is valid and could be exploited
- Fix approach: Remove hardcoded credentials entirely; require environment variables and fail explicitly with clear error message if not present; use separate dev/prod config files if needed

**Duplicate "Old" Codebase:**
- Issue: `src/un_report_apiold/` directory contains complete duplicate of API code with outdated logic
- Files: `src/un_report_apiold/` (entire directory)
- Impact: Maintenance burden; code drift between old and new versions; potential confusion about which implementation is current; unused code adds to deployment size; test coverage split across two codebases
- Fix approach: Delete `src/un_report_apiold/` entirely; ensure all functionality is in `src/un_report_api/`; if keeping for reference, move to separate `legacy/` or documentation

**Bare Pass Statements After Exception Handling:**
- Issue: Multiple nested `try/except` blocks with bare `pass` statements, effectively ignoring exceptions
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1749, 1756, 1763, 1795, 1821, 1827, 1869)
- Impact: Silent failures in fallback element selection logic; web scraper may proceed with wrong state; difficult to diagnose why scraping fails
- Fix approach: Log the exception with context; set a flag indicating element not found; allow graceful degradation with clear error logging

**Circular or Ambiguous Imports:**
- Issue: Multiple `import` and `from` statements mixed; imports appear multiple times (e.g., `import os`, `import json`, `import pandas` repeated)
- Files: `src/un_report_api/app/main.py` (lines 3-55)
- Impact: Code cleanliness issues; potential import order problems; makes dependency graph harder to trace
- Fix approach: Consolidate imports at the top; use one section per category (stdlib, third-party, local); remove duplicate imports

**Path Manipulation Instead of Package Structure:**
- Issue: `sys.path.insert(0, ...)` used to manipulate import paths instead of using proper Python package structure
- Files:
  - `src/un_report_api/app/main.py` (lines 14-16)
  - `src/un_report_api/app/services/comprehensive_veto_regeneration.py` (lines 25-30)
- Impact: Makes code fragile when run from different directories; relies on relative paths; difficult to deploy in containers or different environments; breaks static analysis tools
- Fix approach: Use proper relative imports (`from . import module`); ensure all packages have `__init__.py`; test imports from project root directory

## Known Bugs

**Bare Except Not Logging:**
- Symptoms: Scraper continues after exception at line 1868 without any indication of what failed
- Files: `src/un_data_pipeline/scraper_pipeline.py` (line 1868)
- Trigger: When clearing filters fails after selecting a year, the bare `except:` with only `pass` statement silently suppresses the error
- Workaround: Scraper may retry with fresh driver connection, but root cause is never surfaced

**Parse Error Silent Failure:**
- Symptoms: Dashboard pipeline fails to convert country codes without indication
- Files: `src/un_data_pipeline/dashboard_data_pipeline.py` (line 349)
- Trigger: When parsing tags_flat from CSV field with bare except, invalid format returns None instead of raising
- Workaround: Scraper output must be in exact expected format; no validation of input CSV

**Veto Selection Fallback Logic Fragile:**
- Symptoms: Year selection may fail silently if all three element location strategies (ID, data-value, text search) fail
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1740-1830)
- Trigger: Website UI changes that affect element selectors; missing elements with expected IDs
- Workaround: Fresh driver connection and user-agent rotation after 5 "no such element" errors, but underlying CSS selector issues remain

## Security Considerations

**Exposed API Keys in Code:**
- Risk: Hardcoded JWT token for Supabase in development mode is valid and can be extracted from source code or git history
- Files: `src/un_report_api/app/supabase_client.py` (line 25)
- Current mitigation: Token marked as "development default" with warning log; may have limited permissions
- Recommendations:
  - Never hardcode any credentials; use environment variables exclusively
  - If this token is production-capable, revoke immediately from Supabase console
  - Add pre-commit hook to detect hardcoded secrets (use `detect-secrets` or `git-secrets`)
  - Scan git history: `git log -p | grep -i "key\|token\|password"`

**CORS Misconfiguration:**
- Risk: CORS allows requests from single origin only (`https://datadrivendecisionlab.com`) but no validation of Host header
- Files: `src/un_report_api/app/main.py` (lines 66-72)
- Current mitigation: Specific origin whitelist prevents wildcard CORS
- Recommendations:
  - Ensure origin validation is enforced at load balancer/proxy level
  - Log CORS rejection attempts to detect attack patterns
  - Consider adding rate limiting per origin

**Environment Variable Fallback Logic:**
- Risk: API attempts to function with hardcoded fallback instead of failing fast
- Files: `src/un_report_api/app/supabase_client.py` (lines 19-26)
- Current mitigation: Warning logs are emitted
- Recommendations:
  - Fail immediately with clear error if env vars missing: `raise ValueError("SUPABASE_URL environment variable required")`
  - Never use fallback hardcoded values in production code

## Performance Bottlenecks

**Synchronous Selenium Web Scraping:**
- Problem: Scraper uses synchronous Selenium with hardcoded delays (`time.sleep(0.2)`, `time.sleep(1.5)`) totaling significant latency
- Files: `src/un_data_pipeline/scraper_pipeline.py` (multiple sleep calls, approximately 25+ instances)
- Cause: Web scraping waits for page loads with fixed delays; no event-based waiting except WebDriverWait; nested retry loops with additional delays
- Improvement path:
  - Replace fixed `time.sleep()` with WebDriverWait conditions
  - Consider async scraping framework (Playwright with async/await)
  - Profile actual page load times to set realistic WebDriverWait timeouts
  - Batch process multiple years in parallel with thread pools (already started with ThreadPoolExecutor but blocking on sleep calls)

**DataFrame-Wide Operations in Loop:**
- Problem: `dashboard_data_pipeline.py` applies functions across entire DataFrame in nested groupby operations
- Files: `src/un_data_pipeline/dashboard_data_pipeline.py` (lines 359-369)
- Cause: Calculating alignment scores requires iterating through all tags and all votes; no indexing or caching
- Improvement path:
  - Pre-compute year counts for each tag group before iteration
  - Use vectorized operations where possible instead of iterating rows
  - Consider caching results if same calculations repeated

**CSV Loading Into Memory:**
- Problem: All CSV data loaded entirely into Pandas DataFrames without pagination or streaming
- Files: `src/un_report_api/app/supabase_client.py` (lines 31-155)
- Cause: Uses `pd.read_csv()` without chunking; datasets could grow large
- Improvement path:
  - For large files, use `chunksize` parameter
  - Consider database queries instead of CSV files (use actual Supabase tables)
  - Implement LRU cache for frequently accessed data

**Nested Try/Except in Element Location:**
- Problem: Three nested fallback strategies each with separate try/except blocks and element finding calls
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1745-1763, 1815-1827)
- Cause: Code duplication in element selection with different strategies; each strategy has full overhead
- Improvement path:
  - Extract element finding logic to helper function with strategy list
  - Use single loop with multiple selector strategies
  - Cache element selectors that work

## Fragile Areas

**Web Scraper Selector Dependencies:**
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1740-1839)
- Why fragile: XPath and ID selectors hardcoded based on specific website DOM structure; website redesign breaks all selectors; multiple fallback strategies suggest previous failures
- Safe modification:
  - Add visual regression testing for website changes
  - Use more stable selectors (data attributes rather than generated IDs)
  - Monitor for selector breakage with explicit error logging
  - Keep detailed log of known website selector mappings
- Test coverage:
  - No unit tests for selector logic; integration tests would require live website access
  - Consider creating test fixtures with mock HTML samples

**Veto Data Enhancement Service:**
- Files: `src/un_report_api/app/services/comprehensive_veto_regeneration.py` (lines 73-120)
- Why fragile: Multiple conditional branches on string matching (canonical_topic); if topic naming changes, descriptions become generic; relies on specific CSV field names from DPPA source
- Safe modification:
  - Define topic mappings in configuration file instead of hardcoded strings
  - Add schema validation for DPPA source data
  - Write tests for each topic type description generation
  - Version topic mapping to handle schema changes
- Test coverage: No visible tests for veto enhancement; changes to description generation are untested

**Tag Parsing from CSV Fields:**
- Files: `src/un_data_pipeline/dashboard_data_pipeline.py` (lines 345-357)
- Why fragile: Attempts `ast.literal_eval()` first, falls back to string splitting; assumes specific format in CSV; bare `except:` masks parsing errors
- Safe modification:
  - Validate CSV format at load time with schema
  - Use proper serialization (JSON instead of repr'd lists)
  - Add comprehensive error logging for unexpected formats
  - Create test data with known edge cases (empty, malformed, missing tags)
- Test coverage: No visible tests for tag parsing logic

**Import Path Dependency on Working Directory:**
- Files: `src/un_report_api/app/main.py` (lines 14-16)
- Why fragile: Relative imports work when run from specific directory; breaks when run from project root or in container; depends on `__file__` location being in expected place
- Safe modification:
  - Use absolute imports or proper package structure
  - Test imports from multiple working directories
  - Document expected working directory in README
  - Use `pip install -e .` for development to make package importable
- Test coverage: No clear test for import paths under different working directories

## Scaling Limits

**Single-Threaded Scraper Loop:**
- Current capacity: 25 years of data from UN Digital Library (1946-2024 approximately)
- Limit: Website rate limiting; browser resources; storage for CSV output
- Scaling path:
  - Implement distributed scraping with queue-based architecture
  - Use browser pool (multiple Selenium instances) with proper cleanup
  - Add persistent job queue for failed links/years
  - Consider headless mode optimization or browser service (Browserless API)

**In-Memory Dashboard Scoring:**
- Current capacity: Limited by available RAM for full DataFrame operations
- Limit: Large datasets exceed available system memory during groupby operations
- Scaling path:
  - Migrate from CSV to database (use actual Supabase tables)
  - Implement stream processing for score calculations
  - Use Dask or Spark for distributed computation if datasets grow

**API Endpoint Response Time:**
- Current capacity: Data loaded fresh on each request from CSV files
- Limit: Large CSV files cause slow response; no caching mechanism
- Scaling path:
  - Implement Redis caching for report generation results
  - Pre-compute common queries and store in database
  - Add pagination for large result sets
  - Use CDN for static rankings data

## Dependencies at Risk

**Selenium Version Drift:**
- Risk: `selenium>=4.12.0` specified but ChromeDriver compatibility issues common; webdriver-manager helps but can fail
- Impact: Browser automation breaks; scraper cannot access website
- Migration plan:
  - Consider Playwright instead (better async support, bundled browser management)
  - Pin Selenium version to known-good release
  - Add integration tests that verify Selenium + ChromeDriver compatibility

**OpenAI API Dependency:**
- Risk: API calls for LLM tagging depend on OpenAI service availability and rate limits; no local fallback
- Impact: Tagging fails if API unreachable; costs grow with volume
- Migration plan:
  - Add local LLM option (Ollama, local transformers) as fallback
  - Implement request caching to avoid re-tagging identical content
  - Queue failed tagging requests for retry

**Supabase Client Deprecated Pattern:**
- Risk: Fallback to hardcoded credentials instead of failing fast; code suggests previous Supabase schema changes
- Impact: Development/production confusion; credentials exposure risk
- Migration plan:
  - Fully migrate away from CSV loading to proper Supabase queries
  - Remove all hardcoded credentials
  - Add clear error messages for configuration issues

## Missing Critical Features

**No Logging Infrastructure:**
- Problem: Basic Python logging exists but no centralized log aggregation; logs written to `logs/un_scraper_tagger.log` locally only
- Blocks: Cannot debug production issues; no audit trail for data changes
- Solution: Implement ELK stack (Elasticsearch, Logstash, Kibana) or cloud alternative (CloudWatch, Stackdriver)

**No Error Monitoring:**
- Problem: Bare except blocks mean exceptions are never recorded; no alerting system
- Blocks: Production failures go unnoticed until data inconsistency discovered
- Solution: Add Sentry or similar error tracking; configure alerts for exception thresholds

**No Data Validation Schema:**
- Problem: CSV data not validated against schema; no type checking on loaded data
- Blocks: Garbage data from website scraping silently corrupts database
- Solution: Use Pydantic models for all data validation; add pre-flight validation before CSV load

**No API Rate Limiting:**
- Problem: No rate limiting on endpoints; OpenAI API calls not batched
- Blocks: API can be abused; costs could spike unexpectedly
- Solution: Add FastAPI rate limiting middleware; batch LLM requests with exponential backoff

**No Test Suite:**
- Problem: No visible unit tests, integration tests, or test fixtures
- Blocks: Refactoring is risky; regressions undetected
- Solution: Add pytest suite; start with critical paths (tagging, report generation, CSV parsing)

## Test Coverage Gaps

**Scraper Pipeline Untested:**
- What's not tested: Selenium element selection, year filtering, link extraction, CSV writing
- Files: `src/un_data_pipeline/scraper_pipeline.py` (entire file, 2395 lines)
- Risk: Website UI changes break scraper without detection; data corruption unnoticed
- Priority: High

**Report Generation Untested:**
- What's not tested: Report structure, calculations for pillar scores, ranking logic
- Files: `src/un_report_api/app/report_generator.py` (entire file, 670 lines)
- Risk: Incorrect reports served to users; data integrity issues undetected
- Priority: High

**Veto Tagging Untested:**
- What's not tested: Topic classification, geographic tagging, description generation
- Files: `src/un_report_api/app/services/veto_tagging.py` (550 lines)
- Risk: Poor tagging quality; LLM inconsistency undetected
- Priority: Medium

**Dashboard Scoring Untested:**
- What's not tested: Alignment score calculation, normalization, filtering logic
- Files: `src/un_data_pipeline/dashboard_data_pipeline.py` (838 lines)
- Risk: Incorrect rankings published; methodology changes break silently
- Priority: High

**API Endpoint Integration Untested:**
- What's not tested: Request validation, response format, error handling paths
- Files: `src/un_report_api/app/main.py` (323 lines)
- Risk: API breaks during changes; endpoint contracts not maintained
- Priority: Medium

---

*Concerns audit: 2026-03-18*
