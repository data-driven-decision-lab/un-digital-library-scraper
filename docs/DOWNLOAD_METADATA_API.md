# `GET /download/metadata` — dataset metadata

Companion endpoint to `/download/votes`. Returns Turso connectivity status, row counts and date ranges for the two voting tables, and the most recent scraper run. Designed for "Data current as of …" indicators next to the download button — no need to parse the full CSV to surface basic stats.

## Endpoint

```
GET https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/metadata
```

No query parameters. No body.

## Response

- **Status:** `200 OK` (always — even on internal failures; check `turso.healthy` for the actual state)
- **Content-Type:** `application/json`
- **Body shape:**

```json
{
  "turso": {
    "healthy": true,
    "checked_at": "2026-05-18T16:54:58Z"
  },
  "tables": {
    "un_votes_unga": {
      "rows": 7373,
      "earliest_date": "1946-01-24 00:00:00+00",
      "latest_date": "2026-04-24 00:00:00"
    },
    "un_votes_with_sc": {
      "rows": 10118,
      "earliest_date": "1946-01-24 00:00:00+00",
      "latest_date": "2026-04-30 00:00:00"
    }
  },
  "last_run": {
    "scraper_pipeline": {
      "started_at": "2026-05-18T14:19:12.546536",
      "finished_at": "2026-05-18T14:31:08.722104",
      "status": "success",
      "rows_affected": 24,
      "error_message": null
    }
  }
}
```

## Fields

### `turso`

| Field | Type | Description |
|---|---|---|
| `healthy` | boolean | `true` if the API could open a connection to Turso and run a query. `false` if connectivity failed. |
| `checked_at` | ISO datetime (UTC, `Z`) | Server time when the connectivity check ran. |
| `error` | string (only on failure) | Short description of the connection failure. |

### `tables.<name>`

`<name>` is one of `un_votes_unga`, `un_votes_with_sc`. Each entry:

| Field | Type | Description |
|---|---|---|
| `rows` | integer | Total row count in the table. |
| `earliest_date` | string | Date of the oldest resolution, ISO 8601 (sometimes with `+00` tz suffix). |
| `latest_date` | string | Date of the newest resolution. **Use this for "data through" UI.** |
| `error` | string (only on failure) | Per-table query error message. Other tables still populate. |

Note: dates are filtered server-side to exclude rows where the `Date` column contains the literal string `"nan"` (a data-quality artefact). So `latest_date` is always a real date.

### `last_run.scraper_pipeline`

The most recent scraper run that updated the voting tables. May be `null` if the pipeline has never run.

| Field | Type | Description |
|---|---|---|
| `started_at` | ISO datetime | When the run began. |
| `finished_at` | ISO datetime \| null | When it completed. `null` while still running. |
| `status` | string | `"running"`, `"success"`, or `"failed"`. |
| `rows_affected` | integer | Number of new resolution rows persisted in this run. |
| `error_message` | string \| null | Error traceback if `status = "failed"`. |

## CORS

Allow-list pinned to `https://datadrivendecisionlab.com`. Same policy as the rest of the API.

## Example frontend usage

```ts
type DownloadMetadata = {
  turso: { healthy: boolean; checked_at: string; error?: string };
  tables: Record<'un_votes_unga' | 'un_votes_with_sc', {
    rows: number;
    earliest_date: string;
    latest_date: string;
  } | { error: string }>;
  last_run: {
    scraper_pipeline: null | {
      started_at: string;
      finished_at: string | null;
      status: 'running' | 'success' | 'failed';
      rows_affected: number;
      error_message: string | null;
    };
  };
};

async function getDownloadMetadata(): Promise<DownloadMetadata> {
  const res = await fetch(`${API_BASE_URL}/download/metadata`);
  if (!res.ok) throw new Error(`metadata fetch failed: ${res.status}`);
  return res.json();
}

// Use it next to the download button:
const meta = await getDownloadMetadata();
const through = meta.tables.un_votes_with_sc.latest_date?.slice(0, 10); // "2026-04-30"
const total = meta.tables.un_votes_with_sc.rows;                        // 10118
const lastRefresh = meta.last_run.scraper_pipeline?.finished_at;
const healthy = meta.turso.healthy
             && meta.last_run.scraper_pipeline?.status === 'success';
```

## Suggested UI strings

| Source | Render |
|---|---|
| `tables.un_votes_with_sc.rows` | "10,118 resolutions" |
| `tables.un_votes_with_sc.latest_date.slice(0,10)` | "Data through 2026-04-30" |
| `last_run.scraper_pipeline.finished_at` | "Last refreshed 18 May 2026, 14:31 UTC" |
| `turso.healthy && status==="success"` | Green dot |
| `turso.healthy && status==="running"` | Yellow dot — "Refresh in progress" |
| `!turso.healthy \|\| status==="failed"` | Red dot — "Data may be stale" |

## `curl` (for smoke testing)

```bash
curl -s https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/metadata | jq
```

## Notes

- Cheap to call — runs ~3 small `COUNT/MIN/MAX` queries plus one `pipeline_runs` lookup. Safe to poll every minute if you want a live freshness indicator.
- Always returns `200 OK`. Treat any individual key being absent or carrying an `error` field as a partial failure; the rest of the response is still usable.
- The endpoint only covers the **voting tables** (`un_votes_unga`, `un_votes_with_sc`) that back `/download/votes`. It doesn't include derived tables (`annual_scores`, `topic_votes_yearly`, etc.) because those aren't downloadable here.
