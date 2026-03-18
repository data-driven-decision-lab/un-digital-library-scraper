# Roadmap: UN Digital Library Scraper — Pipeline Revamp & Turso Migration

## Overview

This milestone revamps the pipeline to fix data quality bugs (tag loss, duplicates, stale data, non-voting country inconsistencies), migrates the database from Supabase to Turso (LibSQL), and ships the result with updated documentation and CI/CD. The brownfield codebase is cleaned first to remove dead code and junk data, then the new database layer replaces Supabase, then pipeline logic is corrected on top of the clean foundation, then documentation and deployment are finalized together.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Cleanup** - Remove dead code, junk data, and stale artifacts from the codebase
- [ ] **Phase 2: Database Migration** - Replace Supabase with Turso (LibSQL) across pipeline and API
- [ ] **Phase 3: Pipeline Fixes** - Correct data quality bugs and standardize pipeline behavior
- [ ] **Phase 4: Ship** - Finalize documentation, update CI/CD, and deploy

## Phase Details

### Phase 1: Cleanup
**Goal**: The codebase contains only live, relevant code and clean data references
**Depends on**: Nothing (first phase)
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04
**Success Criteria** (what must be TRUE):
  1. `src/un_report_apiold/` directory no longer exists in the repository
  2. The test resolution `A/RES/79/125` does not appear in any pipeline logic or raw data references
  3. Junk tags (`test`, `data-type-fix`, empty strings) are not produced or consumed by any pipeline function
  4. No unused scripts, stale CSVs, or orphaned files remain that were identified in the audit
**Plans**: TBD

### Phase 2: Database Migration
**Goal**: The pipeline and API read from and write to Turso (LibSQL) exclusively — Supabase is fully removed
**Depends on**: Phase 1
**Requirements**: DB-01, DB-02, DB-03, DB-04, DB-05, DB-06
**Success Criteria** (what must be TRUE):
  1. Turso schema exists with all required tables (un_votes_raw, un_votes_with_sc, annual_scores, topic_votes_yearly, pairwise_similarity_yearly, pipeline_runs)
  2. Running the pipeline writes data to Turso using upsert logic — no delete-then-insert
  3. All API endpoints return data sourced from Turso, not Supabase
  4. Unique constraints on topic_votes_yearly (Year/Country/TopicTag) and pairwise_similarity_yearly (Year/Country1/Country2) prevent duplicate rows at the database level
  5. `pipeline_runs` table records each execution with metadata (start time, status, rows affected)
**Plans**: TBD

### Phase 3: Pipeline Fixes
**Goal**: Pipeline produces accurate, complete, and consistent voting analytics — tag loss, duplicates, and stale data are eliminated
**Depends on**: Phase 2
**Requirements**: PIPE-01, PIPE-02, PIPE-03, PIPE-04, PIPE-05, PIPE-06
**Success Criteria** (what must be TRUE):
  1. `generate_topic_votes()` retains all applicable UNBIS tags (Main Category + Subcategory levels), not just top-level matches — tag coverage is materially higher than the pre-fix baseline of 28/268
  2. topic_votes_yearly contains no duplicate rows after a pipeline run
  3. Countries with zero votes (AFG, VEN) are excluded consistently from all output tables (annual_scores, topic_votes_yearly, pairwise_similarity_yearly)
  4. CosineSimilarity values are stored at full float precision — `round(x, 4)` is not applied
  5. The `abs(x) > 1e3` guard does not silently nullify valid score values
**Plans**: TBD

### Phase 4: Ship
**Goal**: The revamped pipeline is deployed, documented, and reproducible by any team member
**Depends on**: Phase 3
**Requirements**: DOC-01, DOC-02, DOC-03, DOC-04, CICD-01, CICD-02, CICD-03
**Success Criteria** (what must be TRUE):
  1. A methodology document exists explaining P1 4-year rolling window, normalization, and pillar formulas
  2. The Turso schema is documented with table descriptions and column semantics
  3. README contains working setup, architecture, and deployment instructions
  4. Cloud Build / GitHub Action uses Turso environment variables and the pipeline deploys successfully to Cloud Run
  5. `.env.example` lists all required Turso credential placeholders
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Cleanup | 0/TBD | Not started | - |
| 2. Database Migration | 0/TBD | Not started | - |
| 3. Pipeline Fixes | 0/TBD | Not started | - |
| 4. Ship | 0/TBD | Not started | - |
