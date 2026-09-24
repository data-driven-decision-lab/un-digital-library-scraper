# legacy/supabase

Leftovers from the Supabase era of this pipeline, kept for reference only. Nothing in the scraper, the scoring pipeline or the API imports them.

- `supabase_client.py`: the API's old data loader (`SupabaseDataLoader`), which fed annual scores, pairwise similarity and topic votes to the report endpoints. The data moved to Turso in March 2026: `src/un_report_api/app/turso_client.py` replaced this loader, and this file was reduced to a stub that raises `ImportError`.

The Supabase project is offline (its hostname no longer resolves), so this code cannot run. The live database is Turso: see `db/schema.sql` and `docs/SCHEMA.md`.
