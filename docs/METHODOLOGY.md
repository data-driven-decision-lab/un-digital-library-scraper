# Pipeline Methodology

This document explains how the UN Digital Library Dashboard Data Pipeline computes
pillar scores, rankings, and similarity metrics. It is intended for team members who
need to understand or audit the computation without reading source code.

## How to read the scores

All pillar scores are on a **0–100 scale**. Higher is better. Scores are
**relative within each calendar year**: a score of 80 in 2010 and a score of 80
in 2020 both mean the country ranked in the top tier *for that year*, not that its
absolute voting behaviour was identical. Do not compare raw scores across years
unless comparing the underlying ranks.

---

## Section 1: Overview

The pipeline produces three output tables:

| Table | Contents |
|---|---|
| `annual_scores` | Per-country, per-year pillar scores (P1/P2/P3), total index, ranks, and vote counts |
| `topic_votes_yearly` | Per-country, per-year, per-topic vote counts |
| `pairwise_similarity_yearly` | Cosine similarity between every pair of countries for each year |

**Data source:** `un_votes_with_sc` (Turso table).
The pipeline filters out Security Council resolutions at load time: any row whose
`Resolution` field starts with `"S/"` is excluded before all computations.

**Zero-vote exclusion (PIPE-03):** Any country-year combination that cast zero
votes (YES, NO, or ABSTAIN) after SC filtering is excluded from all three output
tables. These rows contain no signal and would distort rankings and similarity scores.

---

## Section 2: Pillar 1 — Policy Consistency Score

### Purpose

Measures how consistently a country votes within its declared policy positions over
a four-year rolling window. A high P1 score indicates stable, predictable voting
behaviour across topic areas.

### Tag parsing

Resolution tags are matched against the UNBIS classification hierarchy
(`un_classification` dictionary). For Pillar 1, the `parse_tags_p1()` function
returns the **Subcategory** tag of the first matching Main Category / Subcategory
pair found in the tag list. For topic vote counting (Section 3A), the
`parse_tags_for_subtag1()` function matches both **Main Category** and
**Subcategory** level tags.

Only UNBIS Main Category and Subcategory tags participate in scoring — Specific
Items are not used.

### Rolling window

`bloc_size_p1 = 4`

For a given target year Y, the **bloc** is the four consecutive years
`[Y-3, Y-2, Y-1, Y]`.

Because three prior years are required to form the first bloc, P1 scores begin
in the fourth year of available data. Years before that window start have no P1
score.

### Consistent-tag filter

Within a bloc, only **tag groups that appear in all 4 years** are used. This
ensures the deviation measurement is computed over a stable, recurring policy area
rather than one-off resolutions.

### Weighted deviation formula

For each consistent tag group T:

1. Compute the **bloc-level vote distribution** across all votes in all 4 years
   combined:

   ```
   avg_pct[v] = (count of outcome v across the full bloc) / total_votes[T]  × 100
   ```

   where `v ∈ {YES, NO, ABSTAIN}`.

2. For each year in the bloc, compute the **yearly deviation**:

   ```
   yearly_raw_deviation = Σ |year_pct[v] − avg_pct[v]|   (sum over YES, NO, ABSTAIN)
   yearly_deviation_normalized = yearly_raw_deviation / 200.0
   ```

   The divisor 200.0 is the theoretical maximum of `yearly_raw_deviation`
   (e.g., 100 % shift from one outcome to another yields |100 − 0| + |0 − 100| = 200).

3. Compute the **weighted deviation** for tag group T:

   ```
   weighted_deviation[T] = mean(yearly_deviations) × total_votes[T]
   ```

   Multiplying by `total_votes[T]` weights each tag group by how many resolutions it
   covers, preventing sparse tag groups from dominating the score.

### Final raw P1 score

```
P1_raw = max(0,  1 − sum(weighted_deviations) / total_votes_consistent_tags)  × 100
```

where `total_votes_consistent_tags` is the total number of votes cast across all
consistent tag groups in the bloc.

### Min-max normalization

The raw P1 score is **min-max normalized per calendar year** to the 0–100 range:

```
P1_Score = 100 × (P1_raw − min_year) / (max_year − min_year)
```

If all countries in a year have identical raw scores the normalized value defaults
to 50.0.

**"Pillar 1 Score" in `annual_scores` IS the normalized value.** The raw
pre-normalization score is not persisted.

---

## Section 3: Pillar 2 — Regional Alignment Score

### Purpose

Measures how closely a country votes with the aggregate position of its UN regional
bloc. A high P2 score indicates strong alignment with the country's region.

### Region mapping

Country-to-region assignments are loaded from
`data/reference/UN_Country_Region_Mapping.csv`. Countries without a mapping entry
are excluded from Pillar 2 computation.

### Sub-metrics

Two sub-metrics are computed per country per year:

**BMM — Bloc Majority Match**

The percentage of resolutions in the year where the country voted with the modal
(majority) vote of its region. Tied regional majorities are excluded from the BMM
denominator.

```
BMM = (# resolutions where country_vote == regional_majority_vote) /
      (# resolutions with a clear regional majority)  × 100
```

**BDS — Bloc Directional Similarity**

Cosine similarity between the country's vote percentage vector and the region's
aggregate vote percentage vector, scaled to 0–100:

```
v_country = [YES%, NO%, ABSTAIN%]   (country's share of votes cast in the year)
v_region  = [YES%, NO%, ABSTAIN%]   (region's aggregate share)

BDS = cosine_similarity(v_country, v_region)  × 100
```

Both vectors use percentage shares (not counts) so that countries with different
numbers of votes are comparable.

### P2 formula

```
P2 = mean(BMM, BDS)
```

If either sub-metric is undefined (e.g., no valid majority resolutions) the country
receives no P2 score for that year.

### Normalization

P2 is min-max normalized per year to 0–100 using the same formula as P1.
**"Pillar 2 Score" in `annual_scores` IS the normalized value.**

---

## Section 4: Pillar 3 — Global Alignment Score

### Purpose

Measures how closely a country votes with the global majority across all UN member
states. A high P3 score indicates high conformity with the international consensus.

### Sub-metrics

**GMMC — Global Majority Match Count**

The percentage of resolutions where the country matched the global majority vote.
Tied global majorities are excluded.

```
GMMC = (# resolutions where country_vote == global_majority_vote) /
       (# resolutions with a clear global majority)  × 100
```

**GDSC — Global Directional Similarity**

Cosine similarity between the country's vote percentage vector and the global
aggregate vote percentage vector, scaled to 0–100:

```
v_country = [YES%, NO%, ABSTAIN%]
v_global  = [YES%, NO%, ABSTAIN%]   (aggregate across all countries in the year)

GDSC = cosine_similarity(v_country, v_global)  × 100
```

### P3 formula

```
P3 = mean(GMMC, GDSC)
```

### Normalization

P3 is min-max normalized per year to 0–100.
**"Pillar 3 Score" in `annual_scores` IS the normalized value.**

---

## Section 5: Total Index

```
Total Index Average = mean(P1_Normalized, P2_Normalized, P3_Normalized)
```

The mean is computed across whichever normalized pillars are non-null for the
country-year. The result is already on the 0–100 scale because it is an average
of three 0–100 values.

```
Total Index Normalized = Total Index Average
```

There is **no second normalization step** — the value is carried over directly.
Both columns hold the same number.

**Overall Rank:** Ranked descending by `Total Index Average` within each year.
Rank 1 = highest score in that year.

**Overall Rank Rolling Avg (3y):** 3-year rolling mean of `Overall Rank` per
country, computed in chronological order with `min_periods=1`.

---

## Section 6: Pairwise Similarity

### Vote encoding

Each country's vote record for a year is encoded as an integer vector over all
resolutions in that year:

| Vote | Encoded value |
|---|---|
| YES | 1 |
| NO | -1 |
| ABSTAIN | 0 |
| null (did not participate) | 0 |

### Cosine similarity

The pairwise cosine similarity between countries A and B for year Y is computed
using scikit-learn's `cosine_similarity` on the full encoded vote matrix
(resolutions × countries transposed to countries × resolutions).

**Zero-vector exclusion:** Countries whose encoded vote vector is all zeros after
the encoding step (i.e., they cast no YES or NO votes in the year) are excluded
before the cosine similarity computation. This matches the PIPE-03 exclusion rule.

### Deduplication

Only pairs where `Country1 < Country2` (lexicographic order) are stored. To query
the similarity between two countries, use the alphabetically lower ISO3 code as
`Country1`.

### Precision

Cosine similarity values are stored at full float precision. No rounding is applied
(the `round(x, 4)` step was removed in Phase 3).

Values range from -1.0 (perfectly opposite voting) to +1.0 (identical voting).
A value of 0 indicates no linear relationship in the encoded vote vectors.
