# `GET /download/votes` — bulk CSV export

New endpoint on the UN Report API that streams UN voting data as CSV. Use it for "Download all votes" buttons or any client-side analysis that needs the full dataset.

## Endpoint

```
GET https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/votes
GET https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/votes?type=unga
GET https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/votes?type=sc
```

That base URL is the same origin the frontend already uses for `/report/{iso}`, `/rankings/{year}`, etc.

## Query parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `type` | `"unga"` \| `"sc"` | `"unga"` | Which chamber to export. `unga` = General Assembly only; `sc` = Security Council only. |

Omit the param to get UNGA data.

## Response

- **Status:** `200 OK`
- **Content-Type:** `text/csv`
- **Content-Disposition:** `attachment; filename="un_votes_unga.csv"` or `un_votes_sc.csv`
- **Body:** Streamed CSV. Hitting the URL in a `<a href>` or `window.open` triggers a native browser download.

### Row counts (as of 2026-05-18)

| `type` | Rows | Approx. size |
|---|---|---|
| `unga` | 7,373 | ~26 MB |
| `sc` | 2,745 | ~9 MB |

Counts grow on every scraper run.

## CSV columns

### `type=unga`

| Column | Type | Notes |
|---|---|---|
| `id` | integer | Auto-increment primary key. Not stable across full refreshes — don't persist client-side. |
| `Resolution` | string | e.g. `A/RES/79/1` |
| `Date` | string | ISO 8601, occasionally with `+00` timezone suffix. Strip with `s.replace(/[+-]\d{2}(:\d{2})?$/, '')` before `new Date()` if needed. |
| `Title` | string | Full resolution title |
| `Link` | string | URL on digitallibrary.un.org |
| `tags` | string | Comma-separated UNBIS subject tags |
| `vote_data` | string | JSON blob — see below |

### `type=sc`

Same columns as above, plus:

| Column | Type | Notes |
|---|---|---|
| `sc_flag` | integer | Always `1` for this export (filtered server-side) |

## `vote_data` column

A JSON string per row. After `JSON.parse(row.vote_data)` you get an object keyed by ISO 3166-1 alpha-3 country code:

```json
{
  "AFG": "YES",
  "ALB": "NO",
  "DZA": "ABSTAIN",
  "AND": null,
  "ARG": "YES",
  ...
}
```

| Value | Meaning |
|---|---|
| `"YES"` | Country voted yes |
| `"NO"` | Country voted no |
| `"ABSTAIN"` | Country abstained |
| `null` | Country did not participate (non-member at the time, or non-voting) |

Approximately 190 countries per blob.

## CORS

The API has CORS configured for `https://datadrivendecisionlab.com`. If you're calling from `localhost:3000` for local dev and getting CORS errors, hit the API from a server-side proxy or temporarily widen the allowed origins server-side.

## Example usage

### Anchor tag — simplest "Download" button

```html
<a href="https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/votes?type=unga"
   download="un_votes_unga.csv">
  Download UNGA votes
</a>
```

The `Content-Disposition` header on the response means the browser downloads rather than navigates, even without the `download` attribute — but including it sets the fallback filename in older browsers.

### `fetch` + Blob (if you need to process before saving)

```ts
async function downloadVotes(type: 'unga' | 'sc' = 'unga') {
  const res = await fetch(`$https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/votes?type=${type}`);
  if (!res.ok) throw new Error(`Download failed: ${res.status}`);
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `un_votes_${type}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}
```

### Parsing in-memory (no download, just analysis)

```ts
import Papa from 'papaparse';

async function loadVotes(type: 'unga' | 'sc' = 'unga') {
  const res = await fetch(`$https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/votes?type=${type}`);
  const csv = await res.text();
  const { data } = Papa.parse(csv, { header: true, dynamicTyping: true });
  return data.map(row => ({
    ...row,
    vote_data: JSON.parse(row.vote_data as string),
  }));
}
```

### `curl` (for smoke testing)

```bash
# Default (unga)
curl -OJ "https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/votes"

# SC
curl -OJ "https://un-digital-library-scraper-250787756338.europe-west1.run.app/download/votes?type=sc"
```

`-OJ` honours the server-side filename from `Content-Disposition`.

## Errors

| Status | Meaning |
|---|---|
| `200` | OK — CSV stream follows |
| `422` | `type` was something other than `unga` or `sc` |
| `503` | Database unavailable (Turso connection failed). Retry. |

## Notes

- The endpoint is read-only and idempotent — safe to call repeatedly.
- Streaming response: the body starts arriving before the full query finishes, so the browser shows a download progress bar from the start.
- Row order: `ORDER BY Date` ascending. Oldest votes first.
- Filtering server-side isn't supported on this endpoint (it's a bulk dump). If you need per-year or per-country slices, use the existing `/report/{iso}` or `/rankings/{year}` endpoints.
- A companion endpoint `GET /download/metadata` returns row counts, latest date, and last-run timestamps without downloading the full CSV. See `DOWNLOAD_METADATA_API.md`.
