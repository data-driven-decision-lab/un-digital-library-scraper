# Pipeline / Scraper / Aggregation — Issues Report

> **Date**: 2026-03-18  
> **Source**: Automated audit of Supabase tables + CSV extracts  
> **Audit script**: `analysis/4C/pipeline_audit.py`  
> **Priority labels**: 🔴 Critical (blocks analysis), 🟠 High (corrupts outputs), 🟡 Medium (misleading results), ⚪ Low (cosmetic / minor)

---

## 🔴 1. Massive Tag Loss in `topic_votes_yearly` Aggregation

**The most critical issue.** The aggregation pipeline is dropping ~90% of tags when going from `un_votes_raw` → `topic_votes_yearly`.

| Metric | Value |
|--------|-------|
| Unique tags in `un_votes_raw.tags` (2025) | **268** |
| Unique tags in `topic_votes_yearly.TopicTag` (2025) | **29** |
| Tags present in raw but **missing** from topic_votes | **240** |

### What's missing

Entire UNBIS Main Categories are absent from `topic_votes_yearly`, including:

- **DISARMAMENT AND MILITARY QUESTIONS** — 38 resolutions tagged in raw, 0 rows in topic_votes
- **DISARMAMENT** — 38 resolutions
- **NUCLEAR DISARMAMENT** — 11 resolutions
- **NUCLEAR WEAPONS** — 3 resolutions
- **ARMS EMBARGO** — 9 resolutions
- **PEACE** — 37 resolutions
- **DECOLONIZATION** — 17 resolutions
- **DIPLOMACY** — 18 resolutions
- **DIPLOMATIC RELATIONS** — 16 resolutions
- **DEVELOPMENT FINANCE** — 29 resolutions
- **ECONOMIC DEVELOPMENT AND DEVELOPMENT FINANCE** — 29 resolutions
- **INSTITUTIONS** — 55 resolutions
- **POLITICAL CONDITIONS** — 55 resolutions
- **SOCIAL CONDITIONS AND EQUITY** — 48 resolutions
- **SUSTAINABLE DEVELOPMENT** — 13 resolutions
- **SCIENCE AND TECHNOLOGY** — 12 resolutions
- **HEALTH** — 10 resolutions
- **CIVIL SOCIETY** — 10 resolutions
- **HUMANITARIAN AID AND RELIEF** — 13 resolutions

...and 221 more (see full list from `pipeline_audit.py` output).

### The 29 tags that DO appear in `topic_votes_yearly` (2025)

These seem to be **Main Category–level** tags only. The hundreds of Subcategory and Specific Item tags from the UNBIS hierarchy are all missing.

### Impact

- **All topic-level analysis for 2025 is incomplete.** Any claim like "votes on Disarmament declined" is unfounded — the data simply isn't there.
- **Year-over-year topic comparisons are invalid** if earlier years had different tag coverage.
- The multi-tag inflation ratio is actually **deflated** — topic_votes sums are 0.6–0.8× annual totals (see Issue #3), when they should be >1× due to multi-tagging.

### Likely cause

The aggregation step appears to filter to only Main Category tags (the top level of the UNBIS hierarchy) and drops all Subcategory and Specific Item tags. Since `un_votes_raw.tags` stores the full flattened list of all hierarchy levels, the aggregation needs to either:
1. Include all hierarchy levels in `topic_votes_yearly`, or
2. Document which level is being aggregated and ensure it's consistent across years.

---

## 🔴 2. Duplicate Rows in `topic_votes_yearly` (2025)

| Metric | Value |
|--------|-------|
| Total rows in `topic_votes_yearly` for 2025 | 8,796 |
| Duplicate `(Country, TopicTag)` pairs | **7,810** (88.8%) |

Almost every row is **exactly duplicated** — same Country, same TopicTag, same vote counts. This means the table has approximately 2× the rows it should.

### Example

```
Country  TopicTag                        Yes  No  Abstain  Total
AGO      COMPREHENSIVE HEALTH SERVICES    1    0    0       1
AGO      COMPREHENSIVE HEALTH SERVICES    1    0    0       1    ← exact duplicate
AGO      HUMAN RIGHTS                     3    0    1       4
AGO      HUMAN RIGHTS                     3    0    1       4    ← exact duplicate
```

### Impact

- Any query against `topic_votes_yearly` that sums or counts rows without deduplication will return 2× the correct value.
- This may also affect Pillar score computation if the scoring pipeline reads from this table.

### Likely cause

The aggregation pipeline is running twice for 2025 data (e.g., insert without checking for existing rows, or a migration that re-ran), or the pipeline inserts once per hierarchy level but collapses to the same tag name.

---

## 🟠 3. `topic_votes_yearly` Undercounts Votes (Deflated Totals)

Because of issues #1 and #2, the topic vote totals are actually **lower** than annual totals — the opposite of what the multi-tagging design should produce:

| Country | `annual_scores` Total | `topic_votes` Sum | Ratio |
|---------|----------------------|-------------------|-------|
| USA | 192 | 152 | 0.79× |
| GBR | 192 | 152 | 0.79× |
| BRA | 192 | 152 | 0.79× |
| CHN | 192 | 152 | 0.79× |
| ARG | 155 | 90 | 0.58× |
| ISR | 145 | 109 | 0.75× |

With only 29 of 268 tags present, many resolutions are not covered by any tag in `topic_votes_yearly`. This means **some resolutions have zero topic-level representation** in the aggregated table.

### Expected behavior

`topic_votes` sums should exceed `annual_scores` totals because each resolution maps to multiple tags. A ratio of ~3–5× would be normal with full tag coverage.

---

## 🟠 4. AFG and VEN — Inconsistent Handling of Non-Voting Countries

Afghanistan and Venezuela cast **zero votes** in 2025 (all 193 resolutions show `null` in `un_votes_raw`).

| Table | AFG present? | VEN present? |
|-------|-------------|-------------|
| `un_votes_raw` (columns) | Yes (all null) | Yes (all null) |
| `annual_scores` (2025) | **No** | **No** |
| `pairwise_similarity_yearly` (2025) | **Yes** | **Yes** |
| `topic_votes_yearly` (2025) | No | No |

### Impact

- `pairwise_similarity_yearly` has 193 countries in 2025, but `annual_scores` only has 191. This creates a **country-set mismatch** across tables.
- Pairwise similarity for AFG/VEN against any country will be meaningless (zero-vector cosine is undefined or 0).
- Historical comparisons that assume 193 countries per year will break for 2025 in `annual_scores`.

### Expected behavior

Either (a) exclude non-voting countries from all tables consistently, or (b) include them in all tables with null/0 scores and flag them.

---

## 🟠 5. BOL Missing from `topic_votes_yearly` but Present in `annual_scores`

Bolivia (BOL) has rows in `annual_scores` for 2025 but **no rows** in `topic_votes_yearly` for 2025.

### Impact

- Any per-country topic breakdown for BOL will silently return empty results.
- Cross-file joins will drop BOL from topic analysis.

### Likely cause

BOL may have reduced participation (only 40 votes in 2025, down from 91 in 2024). If the aggregation pipeline has a minimum threshold or BOL's votes didn't match any of the 29 surviving tags, it would be excluded.

---

## 🟠 6. Junk Tags in `un_votes_raw`

Two entries in the `tags` column are not real UNBIS tags:

| Tag | Count |
|-----|-------|
| `data-type-fix` | 1 resolution |
| `test` | 1 resolution |

These appear to be debugging artifacts left by the tagging pipeline. They should be stripped before aggregation.

### Impact

Minor — these don't propagate to `topic_votes_yearly` (since they're among the 240 dropped tags), but they contaminate any analysis that reads `un_votes_raw.tags` directly.

---

## 🟡 7. Resolution Count Discrepancy: 193 in Raw vs 192 Max in Annual

| Source | 2025 Count |
|--------|-----------|
| `un_votes_raw` rows with `Scrape_Year=2025` | **193** |
| `annual_scores` max `Total Votes in Year` | **192** |

One resolution in the raw table is not counted in any country's annual total. This could be a resolution where every country abstained or didn't vote, or a resolution that was scraped but shouldn't have been included.

### Needs investigation

Which resolution is the 193rd? Is it a duplicate, a withdrawn resolution, or a legitimate vote that the aggregation missed?

---

## 🟡 8. No Minimum Vote Threshold for P1 Score

| Country | P1 Score | Total Votes | Interpretation |
|---------|---------|-------------|----------------|
| STP (São Tomé) | **100.0** | 12 | Misleading "perfect" alignment on tiny sample |
| MDG (Madagascar) | — | 11 | Score on 11 votes is unreliable |
| SSD (South Sudan 2024) | **0.0** | 19 | Floor score on small sample |

Countries voting on <20 resolutions get extreme P1 scores (0 or 100) that are statistical noise, not meaningful signals. There is no minimum vote threshold or confidence weighting in the P1 computation.

### Recommendation

Either (a) set a minimum vote count (e.g., 50) below which P1 is reported as NULL, or (b) add a confidence/reliability column so consumers can filter.

---

## 🟡 9. December Clustering — 80% of 2025 Resolutions in One Month

| Month | Resolutions |
|-------|------------|
| Jan–Nov 2025 | 39 (20%) |
| **Dec 2025** | **154 (80%)** |

This is normal for the UNGA calendar (the regular session runs Sep–Dec), but it means:
- "2025 data" is overwhelmingly December 2025 votes.
- Year-over-year comparisons with years that had different monthly distributions could be confounded by agenda-timing effects.
- Countries that participate in Sep–Nov sessions but miss December could have artificially low vote counts.

This isn't a bug — it's a known UNGA scheduling pattern — but the report should acknowledge it as context.

---

## 🟡 10. `Scrape_Year` Ambiguity Across UNGA Sessions

Resolutions are assigned to `Scrape_Year` based on their vote date. But a single UNGA session (e.g., the 79th session) spans two calendar years (Sep 2024 – Sep 2025). The pipeline assigns based on calendar year, not session. This means:

- A resolution voted in September 2024 (part of the 79th session) goes into `Scrape_Year=2024`.
- A resolution voted in February 2025 (also part of the 79th session) goes into `Scrape_Year=2025`.

The 2025 data starts Jan 15, 2025 and ends Dec 18, 2025. No date/year mismatches were found — all `Date` years match `Scrape_Year`. But users expecting session-level grouping should be aware that the data is calendar-year-grouped.

---

## 🟡 11. P1=0.0 for Argentina with 155 Votes — Edge Case in Computation

Argentina's P1=0.0 in 2025 is the only P1=0 for a country with >20 votes. Its vote breakdown is Y=42, N=84, A=29 — not an even split, but P1 rounds to exactly 0.0.

This suggests the P1 formula may have a floor effect where moderate inconsistency across topics collapses to zero rather than a low-but-nonzero score. **The P1 computation formula should be documented** so that outputs can be independently verified.

### Recommendation

Expose or document the P1 computation formula (including normalization, weighting, and bounds) so downstream consumers can validate scores.

---

## ⚪ 12. Pairwise Similarity — 169 Pairs Below -0.5 in 2025

All 169 involve the USA on one side. The most extreme:

| Pair | Cosine Similarity |
|------|------------------|
| CUB–USA | -0.938 |
| LBN–USA | -0.917 |
| MNG–USA | -0.917 |
| CHN–USA | -0.894 |
| BRA–USA | -0.853 |
| FRA–USA | -0.720 |
| GBR–USA | -0.694 |

These are real (driven by USA voting No on 174/192 resolutions), not a computation error. But the sheer number of extreme-negative pairs is unprecedented and worth flagging as a data quality check: **has the pairwise computation been verified against the raw vote vectors for 2025?**

---

## ⚪ 13. Empty Placeholder Tables

The following tables exist in Supabase but contain no data:

| Table | Rows | Purpose |
|-------|------|---------|
| `sc_votes` | 0 | Security Council individual votes |
| `sc_vetoes` | 0 | Security Council vetoes |
| `model_sc_votes` | 0 | Modeled/predicted SC votes |

Not a bug, but these should either be populated or removed to avoid confusion.

---

## Summary — Priority Action Items

| # | Priority | Issue | Action |
|---|----------|-------|--------|
| 1 | 🔴 | 240/268 tags dropped in aggregation | Fix aggregation to include all UNBIS hierarchy levels (or document which level is used) |
| 2 | 🔴 | 88.8% duplicate rows in topic_votes (2025) | Deduplicate and prevent re-insertion |
| 3 | 🟠 | topic_votes undercounts (0.6–0.8× annual) | Consequence of #1; will resolve when tags are fixed |
| 4 | 🟠 | AFG/VEN in pairwise but not annual_scores | Standardize non-voting country handling across all tables |
| 5 | 🟠 | BOL missing from topic_votes | Investigate why aggregation skipped Bolivia |
| 6 | 🟠 | Junk tags ("test", "data-type-fix") | Strip from `un_votes_raw.tags` |
| 7 | 🟡 | 193 raw resolutions vs 192 max in annual | Identify and reconcile the extra resolution |
| 8 | 🟡 | No minimum vote threshold for P1 | Add threshold or confidence column |
| 9 | 🟡 | P1 formula not documented | Document and expose computation logic |
| 10 | 🟡 | December clustering (80%) | Not a bug — add context to reports |
| 11 | ⚪ | 169 extreme-negative pairwise pairs | Verify computation against raw vectors |
| 12 | ⚪ | Empty SC tables | Populate or remove |
