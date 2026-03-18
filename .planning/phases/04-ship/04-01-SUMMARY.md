---
phase: 04-ship
plan: "01"
subsystem: infra
tags: [cloudbuild, cloud-run, turso, libsql, docker, cicd, env-vars]

requires:
  - phase: 02-database-migration
    provides: Turso/LibSQL integration requiring TURSO_DATABASE_URL and TURSO_AUTH_TOKEN at runtime
  - phase: 03-pipeline-fixes
    provides: Confirmed pipeline correctness requiring libsql-experimental package

provides:
  - Cloud Run deploy step injects TURSO_DATABASE_URL and TURSO_AUTH_TOKEN via --set-env-vars
  - .env.example template documenting all required credentials for local development
  - Dockerfile confirmed: installs libsql-experimental via requirements.txt (no new changes)

affects:
  - 04-ship
  - Deployment operators configuring Cloud Build substitution variables

tech-stack:
  added: []
  patterns:
    - "Cloud Build $$VAR syntax for passing substitution variables to gcloud CLI flags"
    - ".env.example pattern: fake placeholder values with source-of-truth comments per credential"

key-files:
  created:
    - .env.example
  modified:
    - cloudbuild.yaml

key-decisions:
  - "$$TURSO_DATABASE_URL syntax (double-dollar) used in Cloud Build YAML — Cloud Build escapes this to single $ at runtime"
  - "Dockerfile unchanged — libsql-experimental already covered by requirements.txt pip install step from Phase 02-04"
  - ".env.example committed to version control (not gitignored) so it serves as a public template"

patterns-established:
  - "Credential injection pattern: Cloud Build substitution vars -> --set-env-vars -> Cloud Run service env"
  - ".env.example documents all required env vars with source comments; .env stays gitignored"

requirements-completed:
  - CICD-01
  - CICD-02
  - CICD-03

duration: 1min
completed: 2026-03-19
---

# Phase 4 Plan 1: Ship — CI/CD Turso Integration Summary

**Cloud Run deploy step wired to inject TURSO_DATABASE_URL and TURSO_AUTH_TOKEN via Cloud Build substitution vars; .env.example created with Turso, OpenAI, and optional config placeholders**

## Performance

- **Duration:** ~1 min
- **Started:** 2026-03-18T21:26:48Z
- **Completed:** 2026-03-18T21:27:37Z
- **Tasks:** 2
- **Files modified:** 2 (cloudbuild.yaml modified, .env.example created)

## Accomplishments

- cloudbuild.yaml deploy step now passes `--set-env-vars TURSO_DATABASE_URL=$$TURSO_DATABASE_URL,TURSO_AUTH_TOKEN=$$TURSO_AUTH_TOKEN` to Cloud Run, completing the credential injection chain
- .env.example provides a public template with clearly-fake placeholder values and source-of-truth comments for each required credential
- Dockerfile verified — no changes needed; libsql-experimental was already added to requirements.txt in Phase 02-04

## Task Commits

Each task was committed atomically:

1. **Task 1: Update cloudbuild.yaml with Turso env vars in Cloud Run deploy step** - `8ff23d4` (feat)
2. **Task 2: Verify Dockerfile installs libsql-experimental and create .env.example** - `f8482e7` (feat)

## Files Created/Modified

- `cloudbuild.yaml` - Appended `--set-env-vars` and value string with both Turso credentials to the gcloud run deploy step
- `.env.example` - New file: template for TURSO_DATABASE_URL, TURSO_AUTH_TOKEN, API_KEY, LOG_LEVEL, PIPELINE_SOURCE_TABLE with inline source comments

## Decisions Made

- Used `$$TURSO_DATABASE_URL` (double-dollar) in cloudbuild.yaml — Cloud Build's substitution escape syntax; at runtime this resolves to `$TURSO_DATABASE_URL` which Cloud Run passes as an env var to the container
- Dockerfile left unchanged — `pip install -r requirements.txt` already installs libsql-experimental>=0.0.5; adding a duplicate install line would have been an error
- .env.example uses clearly-fake placeholder values (`your-database-name`, `your-turso-auth-token-here`) to prevent accidental credential leakage

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

**External services require manual Cloud Build configuration before deploy will succeed:**

1. Add Cloud Build substitution variables (or Secret Manager bindings) for:
   - `_TURSO_DATABASE_URL` — your Turso database libsql:// URL
   - `_TURSO_AUTH_TOKEN` — your Turso auth token
2. Ensure the Cloud Build service account has permission to set env vars on the Cloud Run service

No .env.example changes needed — it documents what Cloud Run will receive.

## Next Phase Readiness

- cloudbuild.yaml is deployment-ready once Cloud Build substitution variables are configured
- .env.example provides all local developer context for the Turso integration
- Remaining Phase 04-ship tasks can proceed (if any)

---
*Phase: 04-ship*
*Completed: 2026-03-19*
