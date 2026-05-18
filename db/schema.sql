-- LibSQL/Turso schema for UN Digital Library Scraper.
-- Compatible with SQLite DDL.
-- Apply via: cat db/schema.sql | turso db shell unga-datadrivendecisionlab
--
-- Tables:
--   1. un_votes_unga            - General Assembly votes only (no Security Council)
--   2. un_votes_with_sc         - all votes (General Assembly + Security Council)
--   3. annual_scores            - per-country per-year pillar scores
--   4. topic_votes_yearly       - per-country per-year per-topic vote counts
--   5. pairwise_similarity_yearly - country-pair cosine similarity per year
--   6. pipeline_runs            - execution metadata


-- 1. un_votes_unga: General Assembly votes only (Security Council resolutions excluded).
--    Functionally equivalent to: SELECT ... FROM un_votes_with_sc WHERE sc_flag = 0.
--    Maintained as a separate table for convenience of downstream consumers that
--    only want GA voting data.
--    The scraper produces wide-format rows (one row per resolution, country ISO3
--    codes as columns). Rather than enumerating all 190+ country columns in DDL,
--    per-country votes are stored as a JSON blob in vote_data.
--    vote_data format: {"ISO3": "YES"|"NO"|"ABSTAIN"|null, ...}
CREATE TABLE IF NOT EXISTS un_votes_unga (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    Resolution  TEXT,
    Date        TEXT,
    Title       TEXT,
    Link        TEXT UNIQUE,
    tags        TEXT,
    vote_data   TEXT  -- JSON blob of {country_iso3: "YES"|"NO"|"ABSTAIN"|null}
);


-- 2. un_votes_with_sc: enriched voting data including Security Council resolutions.
--    vote_data stores JSON of {country_iso3: "YES"|"NO"|"ABSTAIN"|null}
--    sc_flag = 1 indicates a Security Council resolution.
CREATE TABLE IF NOT EXISTS un_votes_with_sc (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    Resolution  TEXT,
    Date        TEXT,
    Title       TEXT,
    Link        TEXT UNIQUE,
    tags        TEXT,
    vote_data   TEXT,  -- JSON blob of {country_iso3: "YES"|"NO"|"ABSTAIN"|null}
    sc_flag     INTEGER DEFAULT 0
);


-- 3. annual_scores: per-country per-year pillar scores and rankings.
CREATE TABLE IF NOT EXISTS annual_scores (
    id                              INTEGER PRIMARY KEY AUTOINCREMENT,
    Year                            INTEGER NOT NULL,
    Country                         TEXT NOT NULL,
    "Pillar 1 Score"                REAL,
    "Pillar 2 Score"                REAL,
    "Pillar 3 Score"                REAL,
    "Total Index Average"           REAL,
    "Overall Rank"                  INTEGER,
    "Overall Rank Rolling Avg (3y)" REAL,
    "Total Index Normalized"        REAL,
    "Pillar 1 Normalized"           REAL,
    "Pillar 1 Rank"                 INTEGER,
    "Pillar 2 Normalized"           REAL,
    "Pillar 2 Rank"                 INTEGER,
    "Pillar 3 Normalized"           REAL,
    "Pillar 3 Rank"                 INTEGER,
    "Yes Votes"                     INTEGER,
    "No Votes"                      INTEGER,
    "Abstain Votes"                 INTEGER,
    "Total Votes in Year"           INTEGER,
    UNIQUE (Year, Country)
);


-- 4. topic_votes_yearly: per-country per-year per-topic vote counts.
--    UNIQUE constraint on (Year, Country, TopicTag) satisfies DB-05.
CREATE TABLE IF NOT EXISTS topic_votes_yearly (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    Year                INTEGER NOT NULL,
    Country             TEXT NOT NULL,
    TopicTag            TEXT NOT NULL,
    YesVotes_Topic      INTEGER DEFAULT 0,
    NoVotes_Topic       INTEGER DEFAULT 0,
    AbstainVotes_Topic  INTEGER DEFAULT 0,
    TotalVotes_Topic    INTEGER DEFAULT 0,
    UNIQUE (Year, Country, TopicTag)
);


-- 5. pairwise_similarity_yearly: country-pair cosine similarity per year.
--    UNIQUE constraint on (Year, Country1, Country2) satisfies DB-05.
--    Country1 < Country2 is enforced at insert time to avoid duplicate pairs.
CREATE TABLE IF NOT EXISTS pairwise_similarity_yearly (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    Year                INTEGER NOT NULL,
    Country1            TEXT NOT NULL,
    Country2            TEXT NOT NULL,
    CosineSimilarity    REAL,
    UNIQUE (Year, Country1, Country2)
);


-- 6. pipeline_runs: execution metadata for each pipeline run (satisfies DB-06).
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT UNIQUE NOT NULL,
    pipeline_name   TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    status          TEXT NOT NULL DEFAULT 'running',
    rows_affected   INTEGER DEFAULT 0,
    error_message   TEXT,
    notes           TEXT
);
