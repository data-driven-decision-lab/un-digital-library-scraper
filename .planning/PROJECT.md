# UN Digital Library Scraper — Pipeline Revamp & Turso Migration

## What This Is

A data pipeline and REST API that scrapes UN General Assembly voting records, classifies resolutions with UNBIS tags, computes alignment scores (3 pillars), pairwise similarity, and topic-level breakdowns — then serves analytics via FastAPI. Used by datadrivendecisionlab.com for country-level UN voting analysis reports.

## Core Value

The pipeline must produce accurate, complete, and consistent voting analytics across all tables (annual_scores, topic_votes_yearly, pairwise_similarity_yearly) — and serve them reliably via the API.

## Requirements

### Validated

- ✓ Selenium-based scraper extracts UNGA resolutions from UN Digital Library — existing
- ✓ OpenAI-powered classification tags resolutions with UNBIS subjects — existing
- ✓ Dashboard pipeline computes 3-pillar scores, rankings, cosine similarity — existing
- ✓ FastAPI REST API serves reports, rankings, SC analysis — existing
- ✓ Incremental deduplication via token hashing — existing
- ✓ Google Cloud Run deployment via Cloud Build — existing
- ✓ Pydantic schema validation on all API responses — existing

### Active

- [ ] Migrate database from Supabase to Turso (LibSQL)
- [ ] Fix tag loss in topic_votes aggregation (240/268 tags dropped)
- [ ] Fix duplicate rows in topic_votes_yearly (88.8% duplicates)
- [ ] Fix stale pairwise similarity data for 2025
- [ ] Standardize non-voting country handling (AFG/VEN) across all tables
- [ ] Clean junk data (test resolution, debug tags)
- [ ] Remove dead code (un_report_apiold/, unused files)
- [ ] Add pipeline data validation and deduplication safeguards
- [ ] Write methodology documentation
- [ ] Update GitHub Action / Cloud Build to use Turso
- [ ] Improve save_data function (remove precision truncation, add upserts)

### Out of Scope

- Frontend changes — datadrivendecisionlab.com is separate
- New scoring pillars or formula changes — preserve existing methodology
- Security Council data population (sc_votes, sc_vetoes tables) — future work
- Real-time data processing — batch pipeline is sufficient

## Context

**Audit findings:** Colleague performed automated audit (PIPELINE_ISSUES.md, RECOMPUTATION_GUIDE.md) identifying 13 issues across data quality, pipeline logic, and architecture. 2 critical, 4 high, 5 medium, 2 low priority.

**Key pipeline issues:**
- `generate_topic_votes()` drops 89.6% of tags due to `un_classification` dictionary gaps and parser limitations
- `save_data_to_supabase()` non-atomic delete-then-insert causes duplicates
- Pairwise similarity computed from stale data snapshot (BOL, ARG affected)
- `round(x, 4)` truncates cosine similarity precision
- No minimum vote threshold for P1 scores (12-vote countries get P1=100)

**Migration target:** Turso (LibSQL) database at `libsql://unga-datadrivendecisionlab.aws-eu-west-1.turso.io`

**Existing codebase map:** `.planning/codebase/` — 7 documents covering stack, architecture, structure, conventions, testing, integrations, concerns.

## Constraints

- **Database:** Migrate to Turso LibSQL — credentials provided by Hugo
- **Compatibility:** API response schemas must remain backward-compatible (frontend depends on them)
- **Data:** No Supabase access for reads — work from code and existing CSVs only
- **Deployment:** Must maintain Google Cloud Run deployment path

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Turso over Supabase | Team decision (Hugo) — centralize on LibSQL | — Pending |
| Work from code/CSVs only | No Supabase credentials available | — Pending |
| Exclude non-voting countries (Option A) | Simpler, consistent across all tables | — Pending |
| Include Main Category + Subcategory tags | Balance between coverage and granularity | — Pending |

---
*Last updated: 2026-03-19 after initialization*
