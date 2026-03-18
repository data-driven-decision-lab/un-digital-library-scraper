# Requirements: UN Digital Library Scraper — Pipeline Revamp & Turso Migration

**Defined:** 2026-03-19
**Core Value:** Accurate, complete, and consistent voting analytics across all tables — served reliably via the API

## v1 Requirements

### Repo Cleanup

- [x] **CLEAN-01**: Remove `src/un_report_apiold/` directory and all dead code
- [x] **CLEAN-02**: Remove junk test resolution `A/RES/79/125` from raw data references
- [x] **CLEAN-03**: Clean junk tags (`test`, `data-type-fix`, empty strings) from pipeline logic
- [x] **CLEAN-04**: Remove or archive unused files, scripts, and stale CSV artifacts

### Database Migration

- [x] **DB-01**: Create Turso database schema matching current Supabase tables (un_votes_raw, un_votes_with_sc, annual_scores, topic_votes_yearly, pairwise_similarity_yearly)
- [x] **DB-02**: Implement Turso client module replacing Supabase client (libsql connection)
- [ ] **DB-03**: Migrate `save_data_to_supabase()` to Turso with upsert-based writes instead of delete-then-insert
- [ ] **DB-04**: Migrate all API data reads from Supabase to Turso
- [x] **DB-05**: Add unique constraints on key tables (Year/Country/TopicTag for topic_votes, Year/Country1/Country2 for pairwise)
- [x] **DB-06**: Add pipeline_runs metadata table for tracking pipeline execution history

### Pipeline Fixes

- [ ] **PIPE-01**: Fix `generate_topic_votes()` tag parser to handle full UNBIS hierarchy (Main Category + Subcategory levels, not just top-level matches)
- [ ] **PIPE-02**: Add deduplication before insert in topic_votes pipeline (`drop_duplicates` on Year/Country/TopicTag)
- [ ] **PIPE-03**: Standardize non-voting country handling — exclude countries with zero votes from all output tables consistently
- [ ] **PIPE-04**: Remove `round(x, 4)` precision truncation for CosineSimilarity in save function
- [ ] **PIPE-05**: Remove `abs(x) > 1e3` guard or make it column-specific to prevent silent data nullification
- [ ] **PIPE-06**: Rename misleading column `Pillar X Score` to `Pillar X Score (Normalized)` or document the normalization clearly

### Documentation

- [ ] **DOC-01**: Write methodology document explaining pipeline computation (P1 4-year rolling window, normalization, pillar formulas)
- [ ] **DOC-02**: Document Turso database schema with table descriptions and column semantics
- [ ] **DOC-03**: Write comprehensive README with setup, architecture, and deployment instructions
- [ ] **DOC-04**: Add inline docstrings to all major pipeline functions

### CI/CD & Deployment

- [ ] **CICD-01**: Update GitHub Action / Cloud Build configuration to use Turso environment variables
- [ ] **CICD-02**: Update Dockerfile to include Turso/LibSQL dependencies
- [ ] **CICD-03**: Update .env.example with Turso credential placeholders

## v2 Requirements

### Data Quality Enhancements

- **DQ-01**: Add minimum vote threshold for P1 scores (e.g., 50 votes minimum)
- **DQ-02**: Store both raw and normalized pillar scores in annual_scores
- **DQ-03**: Add confidence/reliability column for low-vote-count countries
- **DQ-04**: Populate Security Council tables (sc_votes, sc_vetoes)

### Monitoring

- **MON-01**: Add data validation checks that run automatically after each pipeline execution
- **MON-02**: Add alerting for tag coverage regression across pipeline runs

## Out of Scope

| Feature | Reason |
|---------|--------|
| Frontend changes | datadrivendecisionlab.com is a separate project |
| New scoring pillars | Preserve existing methodology; formula changes are research decisions |
| Real-time processing | Batch pipeline sufficient for annual UNGA data |
| OAuth/auth on API | Currently unauthenticated by design (public data) |
| SC table population | Separate initiative, not blocking this milestone |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-01 | Phase 1 | Complete |
| CLEAN-02 | Phase 1 | Complete |
| CLEAN-03 | Phase 1 | Complete |
| CLEAN-04 | Phase 1 | Complete |
| DB-01 | Phase 2 | Complete |
| DB-02 | Phase 2 | Complete |
| DB-03 | Phase 2 | Pending |
| DB-04 | Phase 2 | Pending |
| DB-05 | Phase 2 | Complete |
| DB-06 | Phase 2 | Complete |
| PIPE-01 | Phase 3 | Pending |
| PIPE-02 | Phase 3 | Pending |
| PIPE-03 | Phase 3 | Pending |
| PIPE-04 | Phase 3 | Pending |
| PIPE-05 | Phase 3 | Pending |
| PIPE-06 | Phase 3 | Pending |
| DOC-01 | Phase 4 | Pending |
| DOC-02 | Phase 4 | Pending |
| DOC-03 | Phase 4 | Pending |
| DOC-04 | Phase 4 | Pending |
| CICD-01 | Phase 4 | Pending |
| CICD-02 | Phase 4 | Pending |
| CICD-03 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 23 total
- Mapped to phases: 23 (across 4 phases)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-19*
*Last updated: 2026-03-19 — traceability updated to 4-phase structure (DOC + CICD merged into Phase 4)*
