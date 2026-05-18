# Codebase Concerns

**Analysis Date:** 2026-05-18

## Tech Debt

**Supabase Deprecation Artifacts:**
- Issue: `supabase_client.py` exists as deprecated module that raises ImportError on import. Migration to Turso incomplete at symbolic level.
- Files: `src/un_report_api/app/supabase_client.py`
- Impact: Code clarity issue; imports can fail if any lingering references exist (though all production imports have been redirected to `turso_client.py`). Increases maintenance burden.
- Fix approach: Delete file entirely and scan for any remaining imports. The module serves no purpose post-migration.

**HTML Parsing Fragility:**
- Issue: `extract_vote_data_from_html()` uses BeautifulSoup with brittle selectors. Relies on exact class names and structure that can break if UN Digital Library redesigns HTML.
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1292-1356)
- Impact: Scraper fails silently if page structure changes. Votes/metadata may be missing without warning. BeautifulSoup parsing has no fallback if expected DOM elements don't exist.
- Fix approach: Add CSS/selector versioning, implement DOM structure validation before parsing, add screenshot-on-failure logging for debugging scraper breaks.

**Bare Exception Handlers:**
- Issue: Multiple `except Exception` blocks silently swallow errors. Examples:
  - Line 1350: `except Exception as e: logger.debug(...); continue` in metadata parsing loop
  - Line 2006: `except Exception: pass` in filter clearing
  - Line 1922: `except Exception as e: logger.error(...); pass` 
- Files: `src/un_data_pipeline/scraper_pipeline.py`
- Impact: Unknown errors are logged but not propagated. Debugging is hard. Silent failures in Selenium operations can leave browser in inconsistent state.
- Fix approach: Replace bare `Exception` handlers with specific exception types (TimeoutException, NoSuchElementException, StaleElementReferenceException). Re-raise non-recoverable errors.

**Inadequate Error Context:**
- Issue: `process_resolution()` returns `None` on failure without distinguishing between timeout, missing data, and HTML parsing errors.
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1368-1490)
- Impact: Batch failures are recorded as "failed" with no root cause. Operator can't distinguish between "page didn't load" vs "page loaded but votes missing."
- Fix approach: Return error objects with reason codes (TIMEOUT, PARSE_FAILED, DATA_MISSING, etc.) instead of None/boolean.

**Undocumented API Formula:**
- Issue: P1 score computation uses undocumented formula. PIPELINE_ISSUES.md flags this as Issue #9 — "P1 formula not documented."
- Files: `src/un_data_pipeline/dashboard_data_pipeline.py` (lines 421-450: `calculate_alignment_score_p1()`)
- Impact: P1 scores cannot be verified or validated by consumers. Argentina's P1=0.0 edge case (Issue #11) suggests computation has bounds/floor effects not disclosed.
- Fix approach: Document formula explicitly in function docstring with mathematical notation. Expose intermediate computation values (deviations, weights) in a `Pillar1Metadata` table.

---

## Known Bugs

**Duplicate Rows in topic_votes_yearly (2025):**
- Symptoms: 88.8% of rows in `topic_votes_yearly` for 2025 are exact duplicates (same Country, TopicTag, vote counts).
- Files: Data generation in `src/un_data_pipeline/dashboard_data_pipeline.py`; affects aggregation pipeline.
- Trigger: Aggregation logic runs without deduplication. Likely caused by re-insertion on migration or double-processing of hierarchy levels.
- Workaround: Query with `DISTINCT (Country, TopicTag)` to deduplicate on read. Not a proper fix.
- Fix: Investigate aggregation pipeline logic. Add `UNIQUE` constraint to (Country, TopicTag, Year) tuple in Turso schema. Regenerate table from clean `un_votes_raw`.

**Tag Loss in Aggregation (240/268 tags missing):**
- Symptoms: Only 29 of 268 UNBIS tags from `un_votes_raw` appear in `topic_votes_yearly` for 2025. Entire categories like DISARMAMENT, PEACE, DECOLONIZATION are absent.
- Files: Aggregation logic in dashboard pipeline; tag filtering not exposed in code.
- Trigger: Aggregation step filters to only Main Category level and drops Subcategory/Specific Item tags.
- Workaround: None — data is permanently lost in aggregation.
- Fix: Modify aggregation to preserve full UNBIS hierarchy or document which level is selected. Regenerate `topic_votes_yearly` with full tag coverage.

**topic_votes_yearly Undercounts (0.6–0.8× annual totals):**
- Symptoms: Vote sums in `topic_votes_yearly` are 0.6–0.8× the country's annual total. Expected ratio is >1× due to multi-tagging.
- Files: Consequence of tag loss above; affects all downstream analyses that sum topic votes.
- Trigger: Reduced tag coverage means some resolutions are unrepresented in topic aggregates.
- Workaround: None for accurate analysis.
- Fix: Resolves when issue #1 (tag loss) is fixed. Regenerate aggregates after restoring full tag coverage.

**Afghanistan & Venezuela Inconsistency Across Tables:**
- Symptoms: AFG and VEN (non-voting countries in 2025) appear in `pairwise_similarity_yearly` but NOT in `annual_scores` for 2025. Creates country-set mismatch (193 vs 191 countries).
- Files: Data generation in dashboard pipeline; affects cross-table joins.
- Trigger: Inconsistent filtering logic. Non-voting countries are included in pairwise but excluded in annual scores.
- Workaround: Filter manually to 191 countries when joining tables for 2025.
- Fix: Standardize handling across all tables. Either (a) exclude AFG/VEN from all tables consistently, or (b) include them in all tables with null/0 scores and flag them.

**Bolivia Missing from topic_votes_yearly:**
- Symptoms: BOL has rows in `annual_scores` for 2025 but zero rows in `topic_votes_yearly` for 2025.
- Files: Data generation in dashboard pipeline aggregation.
- Trigger: BOL's 40 votes in 2025 may not match any of the 29 surviving tags, or aggregation pipeline has undocumented minimum threshold.
- Workaround: Include BOL only in regional/annual analyses, exclude from topic breakdowns.
- Fix: Investigate aggregation filtering. Likely resolves when tag loss is fixed.

**Junk Tags in Raw Data:**
- Symptoms: Two non-UNBIS tags exist in `un_votes_raw.tags`: `data-type-fix` and `test`. These are debugging artifacts.
- Files: Data in Turso `un_votes_raw` table.
- Trigger: Manual data entry or testing code that wasn't cleaned before ingestion.
- Workaround: Filter these tags in analysis queries.
- Fix: Clean `un_votes_raw` to remove non-UNBIS tags. Add validation to prevent tagging pipeline from inserting invalid tags.

**Resolution Count Discrepancy (193 vs 192):**
- Symptoms: `un_votes_raw` has 193 resolutions for 2025, but annual_scores max total is 192. One resolution is unaccounted for.
- Files: Data in Turso tables `un_votes_raw` and `annual_scores`.
- Trigger: Unclear — could be duplicate, withdrawn, or legitimately aggregated differently.
- Workaround: None.
- Fix: Identify the 193rd resolution. Determine if it should be included or excluded from analysis.

---

## Security Considerations

**API Key Exposure Risk in Scraper:**
- Risk: Gemini API key loaded via `os.getenv("GEMINI_API_KEY")` at module import time. If .env file is committed or log file captures API calls, key could leak.
- Files: `src/un_data_pipeline/scraper_pipeline.py` (line 110: `API_KEY = os.getenv("GEMINI_API_KEY")`)
- Current mitigation: .env files are in .gitignore. API key is not logged directly (only function calls are logged).
- Recommendations: 
  - Add environment variable validation at startup with clear error if missing.
  - Consider secrets manager (e.g., AWS Secrets Manager, HashiCorp Vault) for production.
  - Never log full API responses that might contain request details.

**Turso Auth Token Handling:**
- Risk: `TURSO_AUTH_TOKEN` passed as plaintext in urllib.request headers in `turso_http.py`. Token is visible in HTTP Authorization header.
- Files: `src/un_data_pipeline/turso_http.py` (line 60-62)
- Current mitigation: HTTPS is enforced (URL is converted from libsql:// to https://). Token is not logged.
- Recommendations:
  - Verify HTTPS enforcement is guaranteed (it is).
  - Rotate auth tokens regularly.
  - Monitor Turso audit logs for unauthorized access.
  - Document that this HTTP client is Windows-only (Python import fallback) and should not be used in production if libsql-experimental is available.

**No Input Validation on Query Parameters:**
- Risk: API endpoints accept year, country, and other parameters without strict validation beyond type checking. Potential for injection attacks if Turso client doesn't escape strings properly.
- Files: `src/un_report_api/app/main.py`; request handlers in various endpoints.
- Current mitigation: Year parameters are constrained by Pydantic `ge`/`le` bounds. Country codes are not validated against a whitelist.
- Recommendations:
  - Add country code validation (verify against ISO 3166-1 alpha-2 list).
  - Use parameterized queries (already done in Turso client, good).
  - Add input length limits to prevent extremely large queries.

---

## Performance Bottlenecks

**Scraper: Sequential Processing of Large Batches:**
- Problem: `batch_scrape_resolutions()` processes links sequentially with only 0.2s delay between requests. On high-latency networks, this is slow. Parallel scraping exists but uses ThreadPoolExecutor which may not help with Selenium I/O.
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1492-1533)
- Cause: Selenium operations are I/O-bound; threading helps only if driver operations release the GIL. Current 2-worker default is conservative.
- Improvement path: 
  - Benchmark optimal worker count (may be 4-8 for typical network latency).
  - Add configurable worker count and batch size parameters.
  - Consider asyncio + Playwright instead of Selenium for better concurrency.

**Tag Tagging API Calls Not Rate-Limited:**
- Problem: `get_tags_sequential()` and `tag_new_rows()` make multiple LLM API calls (up to 3 per resolution for staged classification) without rate-limiting to Gemini API quotas. Exceeding quota causes hard failure.
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1100-1165, 1198-1227)
- Cause: `execute_api_call()` has retries with exponential backoff but no request queue or global rate limit. Parallel processing (if enabled) could spike request rate.
- Improvement path:
  - Implement token bucket or sliding window rate limiter.
  - Add request queue with max concurrent API calls.
  - Expose quota metrics to operator (calls remaining, reset time).
  - Add fallback to simpler single-stage classification if quota exhausted.

**DataFrame Operations Without Indexing:**
- Problem: Dashboard pipeline loads entire vote dataset into memory and performs repeated `.apply()` and `.melt()` operations. No indexing on frequently-filtered columns (Country, Year, TagGroup).
- Files: `src/un_data_pipeline/dashboard_data_pipeline.py` (lines 421-534 in pillar analysis functions)
- Cause: Pandas is used for convenience; no columnar index optimization.
- Improvement path:
  - Profile to identify hottest loops (likely Pillar 1 & 2 calculations).
  - Use Pandas `.set_index()` on (Country, Year) before groupby operations.
  - Consider Polars or DuckDB for larger-than-memory datasets.

**HTML Parsing Every Resolution Page:**
- Problem: `extract_vote_data_from_html()` runs BeautifulSoup parse on every resolution, even if page structure is identical. No caching of parsed schema.
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1292-1356)
- Cause: No assumption about stable page structure; re-parsing every page is safest approach but redundant if schema never changes.
- Improvement path:
  - Cache parsed schema structure after first success.
  - Fall back to cache if page structure changes are detected.
  - Profile to confirm parsing is actual bottleneck vs. network I/O.

---

## Fragile Areas

**Selenium WebDriver Initialization & User-Agent Rotation:**
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1229-1260, 275-277)
- Why fragile: `get_driver()` creates new Chrome instance with hardcoded user-agent rotation. No error recovery if Chrome binary is missing or fails to start. User-agent list is inline and never updated.
- Safe modification:
  - Add pre-flight check for Chrome binary existence.
  - Move user-agent list to configuration file.
  - Add timeout and retry logic to WebDriver initialization.
- Test coverage: No unit tests for driver initialization. Scraper fails hard if Chrome is unavailable.

**Year Selection Logic in Collector:**
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 1780-1894: `select_year_with_fallback()`)
- Why fragile: Multiple fallback strategies (ID lookup, data-value, XPath text match) suggest UI structure is fragile. Function tries 5 different element lookup methods but all could fail if DOM changes.
- Safe modification:
  - Add visual regression testing (screenshot comparison) to detect UI changes early.
  - Maintain versioned mapping of year selector IDs in external config file.
  - Add feedback loop to operator when fallbacks are used (log clearly).
- Test coverage: No automated tests. Manual verification only.

**Bare Exception Handlers in Critical Loops:**
- Files: `src/un_data_pipeline/scraper_pipeline.py` (multiple locations)
- Why fragile: `except Exception as e: pass` in loops can hide Selenium crashes, leaving browser in undefined state. Loop continues as if nothing happened.
- Safe modification:
  - Replace with specific exception types.
  - Add state validation check after exception handlers (is driver still responsive?).
  - Break loop on non-recoverable errors instead of silently continuing.
- Test coverage: No tests for error recovery behavior.

**Pydantic Models for LLM Response Parsing:**
- Files: `src/un_data_pipeline/scraper_pipeline.py` (lines 288-294: LocationClassifications, ResolutionTarget)
- Why fragile: LLM response parsing with Pydantic is strict. Any deviation in JSON structure causes parse failure. Fallback returns dummy "error" status instead of escalating.
- Safe modification:
  - Use lenient parsing with `ValidationError` handling.
  - Log full LLM response before parse failure for debugging.
  - Implement graceful degradation (use simple heuristics if LLM parsing fails).
- Test coverage: No tests for LLM parsing failures.

---

## Scaling Limits

**Memory Usage in Dashboard Pipeline:**
- Current capacity: Full vote dataset (all countries, all years) loaded into memory as single DataFrame.
- Limit: Estimated 200K+ rows × 200 country columns = millions of cells. Breaks on systems with <4GB RAM.
- Scaling path:
  - Use chunked processing (year-by-year or country-by-country).
  - Switch to columnar database (Turso supports this; consider direct SQL aggregation).
  - Move aggregations to database queries instead of pandas operations.

**Parallel Scraper Concurrency:**
- Current capacity: 2 workers (hardcoded `DEFAULT_MAX_WORKERS = 2`). Each worker manages one Selenium instance.
- Limit: 2 concurrent pages; scales poorly. Cannot parallelize tagging and scraping simultaneously.
- Scaling path:
  - Increase max workers based on network capacity (test with 4, 8).
  - Use async scraping (Playwright, aiohttp) instead of threading.
  - Separate scraping from tagging — run tagging batch after scraping completes.

**API Endpoint Throughput:**
- Current capacity: Single FastAPI instance; no load balancing documented.
- Limit: Unknown — depends on hardware. Dashboard pipeline may block API if run on same server.
- Scaling path:
  - Run scraper/pipeline on separate worker thread or schedule (Celery, APScheduler).
  - Add caching (Redis) for annual_scores, topic_votes (read-heavy).
  - Horizontal scaling with multiple FastAPI replicas behind load balancer.

---

## Dependencies at Risk

**Selenium WebDriver Management:**
- Risk: Selenium 4.x is stable but WebDriver binary (ChromeDriver) must match Chrome version. Mismatch causes hard failures. `webdriver-manager` package handles this but adds external dependency.
- Impact: Scraper breaks if Chrome is updated without redownloading driver. No fallback to alternative browsers (Firefox, Edge).
- Migration plan: Consider Playwright (cross-browser, better async support, active maintenance). Would require rewrite of Selenium code.

**Gemini API Dependency (for Tagging):**
- Risk: Gemini API pricing and availability. Rate limits could halt scraping. No fallback to other LLMs.
- Impact: Tagging pipeline blocks on API quota exhaustion. No graceful degradation.
- Migration plan: Add OpenAI GPT-4 as fallback. Implement model abstraction layer to support multiple providers.

**BeautifulSoup HTML Parsing:**
- Risk: Stable library but page structure brittleness is real. UN Digital Library redesign would break scraper.
- Impact: Silent data loss (votes not extracted). Hard to debug.
- Migration plan: Add API-based access to UN voting data if available (reduces scraping risk). Otherwise, add DOM structure versioning and alerts.

**LibSQL / Turso Database:**
- Risk: Turso is relatively new (SQLite fork). Long-term viability unclear. HTTP client fallback on Windows is necessary but adds complexity.
- Impact: Windows deployments depend on HTTP client; direct libsql connections not available.
- Mitigation: HTTP client is functional and tested. Turso is backed by ChiselStrike; reasonable stability.
- Plan: Keep libsql-experimental import optional to support Windows CI/CD environments.

---

## Missing Critical Features

**No Automated Health Checks:**
- Problem: Scraper runs but has no mechanism to detect if data quality degraded. A regression in HTML parsing would silently produce empty votes for days before operator noticed.
- Blocks: Data integrity monitoring, automated alerting.
- Recommendation: Add post-scrape validation checks (compare row counts to previous run, flag missing votes, verify tag distribution).

**No Version Control for Turso Schema:**
- Problem: Database schema (table definitions, constraints) is not tracked in git. Manual changes to schema aren't documented.
- Blocks: Rollback, schema migration safety, schema versioning.
- Recommendation: Use database migration tool (Alembic, Flyway) or document schema in SQL migration files committed to git.

**No Audit Trail for Data Changes:**
- Problem: No way to track when records were modified, by whom, or why. Data corrections are not versioned.
- Blocks: Compliance, debugging data discrepancies.
- Recommendation: Add `created_at`, `updated_at`, and optional `change_reason` columns to key tables.

**No Structured Logging for Debugging:**
- Problem: Logs are formatted as plain text. Difficult to parse, aggregate, or alert on in production.
- Blocks: Centralized log aggregation, structured alerting.
- Recommendation: Switch to JSON logging (structlog or python-json-logger). Include trace IDs for request correlation.

---

## Test Coverage Gaps

**No Tests for Scraper:**
- What's not tested: HTML extraction, Selenium operations, error recovery, retry logic.
- Files: `src/un_data_pipeline/scraper_pipeline.py` (entire module)
- Risk: Scraper changes break silently. Regressions in page parsing are not caught.
- Priority: HIGH — scraper is critical path.

**No Tests for Dashboard Pipeline Aggregation:**
- What's not tested: Tag filtering logic, duplicate handling, country filtering, pillar score computation.
- Files: `src/un_data_pipeline/dashboard_data_pipeline.py`
- Risk: PIPELINE_ISSUES.md identified multiple bugs (#1-3, #5) that could have been caught by unit tests.
- Priority: HIGH — aggregation has known bugs.

**No Tests for API Endpoints:**
- What's not tested: Request validation, response format, error handling, caching behavior.
- Files: `src/un_report_api/app/main.py` and endpoint handlers
- Risk: API changes break clients. Parameter validation gaps silently pass invalid inputs to database.
- Priority: MEDIUM — API is user-facing but relatively simple.

**No Integration Tests:**
- What's not tested: End-to-end scraping → aggregation → API flow.
- Risk: Component-level tests might pass but integration fails.
- Priority: MEDIUM — high confidence in individual components but integration unknowns remain.

**No Performance Tests:**
- What's not tested: Scraper throughput, API response times under load, memory usage with full dataset.
- Risk: Performance bottlenecks discovered only in production.
- Priority: MEDIUM — known bottlenecks but no measurements.

---

## Data Quality Gaps

**No Validation on Vote Data:**
- Issue: No checks for impossible values (e.g., country votes on resolution that wasn't voted on, vote != YES/NO/ABSTAIN/null).
- Impact: Corrupted votes silently propagate through aggregation.
- Fix: Add data validation after scraping and before database insert.

**No Deduplication on Raw Scrape:**
- Issue: If scraper reruns same resolution, duplicates are inserted without collision detection.
- Impact: Duplicate rows inflate vote counts.
- Fix: Add `UNIQUE` constraint on resolution link/token in `un_votes_raw`.

**P1 Score Edge Cases Not Handled:**
- Issue: Argentina's P1=0.0 suggests formula has floor effects or edge case behavior undocumented.
- Impact: Scores appear meaningful when they may be statistical artifacts.
- Fix: Document P1 formula explicitly and test edge cases.

---

## Incomplete Migration from Supabase to Turso

**Deprecated Module Still Present:**
- Issue: `supabase_client.py` raises ImportError; kept "temporarily" but migration complete.
- Files: `src/un_report_api/app/supabase_client.py`
- Fix: Delete file. Scan codebase for any remaining imports.

**References to Supabase in Comments:**
- Issue: Code comments reference "Supabase-native approach" or "remove old Supabase logic" suggesting migration is recent and may have stragglers.
- Files: `src/un_data_pipeline/scraper_pipeline.py` (line 1290), dashboard_data_pipeline.py
- Fix: Clean up comments. Review migration completeness.

**HTTP Client Fallback Not Fully Documented:**
- Issue: turso_http.py provides HTTP fallback for Windows but coupling is implicit (try import libsql, fall back to HTTP).
- Files: `src/un_data_pipeline/turso_http.py`, scraper_pipeline.py (lines 50-54), dashboard_data_pipeline.py (lines 16-19)
- Fix: Document fallback strategy explicitly. Add configuration flag to choose client explicitly.

---

*Concerns audit: 2026-05-18*
