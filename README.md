# UN Digital Library Scraper

Pipeline for scraping, classifying, and analyzing UN General Assembly voting data from the UN Digital Library. Data is stored in Turso (LibSQL) and exposed via a FastAPI REST API deployed to Google Cloud Run.

## Architecture

The system has three layers:

1. **Scraper Pipeline** (`src/un_data_pipeline/scraper_pipeline.py`) — Selenium scrapes voting records from [digitallibrary.un.org](https://digitallibrary.un.org). OpenAI GPT-4o-mini classifies each resolution using UNBIS subject tags and geographic tags. Results are written to the `un_votes_unga` (General Assembly only) and `un_votes_with_sc` (all votes, GA + SC) tables in Turso.

2. **Dashboard Pipeline** (`src/un_data_pipeline/dashboard_data_pipeline.py`) — Reads `un_votes_with_sc`, computes Pillar 1/2/3 scores and pairwise cosine similarity, and writes results to three Turso tables (`annual_scores`, `topic_votes_yearly`, `pairwise_similarity_yearly`). CSV copies are saved to `src/un_report_api/app/required_csvs/` as API fallbacks.

3. **REST API** (`src/un_report_api/app/main.py`) — FastAPI serving country reports, rankings, and Security Council analysis. Reads from Turso tables with CSV fallback.

**Data flow:**
```
UN Digital Library → scraper → Turso → dashboard pipeline → Turso + CSVs → API → clients
```

See [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for pillar score formulas and computation details.
See [docs/SCHEMA.md](docs/SCHEMA.md) for the full database table reference.

## Requirements

- Python 3.11+
- Chrome browser (for Selenium scraper)
- [Turso](https://turso.tech) account with a database created
- OpenAI API key (for scraper classification)
- Google Cloud project (for deployment)

## Local Setup

```bash
# Clone repository
git clone https://github.com/data-driven-decision-lab/un-digital-library-scraper.git
cd un-digital-library-scraper

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and fill in:
#   TURSO_DATABASE_URL  — from Turso dashboard (libsql://your-database-name.turso.io)
#   TURSO_AUTH_TOKEN    — from Turso dashboard
#   API_KEY             — OpenAI API key
```

### Initialise the database schema

```bash
# Apply schema to your Turso database
cat db/schema.sql | turso db shell <your-database-name>
```

The full DDL is documented in [docs/SCHEMA.md](docs/SCHEMA.md).

## Running the Pipelines

```bash
# 1. Run the scraper (fetches new UN resolutions, classifies them, writes to Turso)
python -c "import sys; sys.path.insert(0, 'src'); from un_data_pipeline.scraper_pipeline import main; main()"

# 2. Run the dashboard scoring pipeline (reads Turso, computes scores, writes back to Turso)
python -m src.un_data_pipeline.dashboard_data_pipeline
```

The scraper requires Chrome and network access to `digitallibrary.un.org`. Each pipeline run logs execution metadata to the `pipeline_runs` Turso table.

## Running the API Locally

```bash
cd src/un_report_api/app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation is available at `http://localhost:8000/docs` (Swagger UI).

## API Endpoints

### `GET /report/{country_iso}`

Provides a detailed report on a country's UN voting patterns, scores, and regional context.

**Path parameters:**
- `country_iso` (string, required): 3-letter uppercase ISO 3166-1 alpha-3 code (e.g., `USA`).

**Query parameters:**
- `start_year` (integer, required): Start year for the report period.
- `end_year` (integer, required): End year for the report period.

**Response includes:**
- `report_metadata`: Country ISO, name, and period.
- `world_average_scores_period`: World average pillar scores for the period.
- `country_average_scores_period`: Country pillar scores, ranks, and total index average.
- `index_score_analysis`: Index score and rank at start/end of period with percentage change.
- `voting_behavior_overall`: Country vote percentages (yes/no/abstain) vs world averages.
- `most_aligned_p5_member` / `least_aligned_p5_member`: The P5 Security Council member most and least aligned with the country by cosine similarity.
- `scores_timeseries`: Yearly breakdown of pillar scores, ranks, and world averages.
- `top_allies` / `top_enemies`: Top 5 most and least aligned countries by average similarity.
- `top_supported_topics` / `top_opposed_topics`: Top 3 topics by voting direction.
- `all_topic_voting`: Detailed per-topic vote stats vs world averages.
- `regional_context`: Regional alignment for the `end_year`, including peer alignment scores sorted descending.

### `GET /rankings/{year}`

Returns pillar score rankings for all countries in a given year.

### `GET /sc/veto_analysis`

Security Council veto usage analysis.

### `GET /sc/vote_analysis`

Security Council vote distribution analysis.

## Deployment (Google Cloud Run)

Deployment is automated via Cloud Build on push to `main`. The `cloudbuild.yaml` pipeline:

1. Builds a Docker image tagged with the commit SHA.
2. Pushes the image to Google Container Registry (`gcr.io/$PROJECT_ID/unreportapi`).
3. Deploys to the Cloud Run service `unreportapi` in `europe-west1`.

**Required Cloud Build configuration:**

Set `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` as substitution variables in your Cloud Build trigger (use `$$` prefix in `cloudbuild.yaml` to escape Cloud Build variable expansion — the variables resolve to single `$` at runtime).

**Manual deploy:**

```bash
# Build and push image
docker build -t gcr.io/YOUR_PROJECT_ID/unreportapi:latest .
docker push gcr.io/YOUR_PROJECT_ID/unreportapi:latest

# Deploy to Cloud Run
gcloud run deploy unreportapi \
  --image gcr.io/YOUR_PROJECT_ID/unreportapi:latest \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars TURSO_DATABASE_URL=your_url,TURSO_AUTH_TOKEN=your_token
```

The Dockerfile exposes port 8080 and runs `uvicorn` with `--proxy-headers`. Cloud Run injects `TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN` at runtime — `API_KEY` is not required by the API (only by the scraper pipeline).

## Project Structure

```
un-digital-library-scraper/
├── src/
│   ├── un_data_pipeline/
│   │   ├── scraper_pipeline.py        # Web scraper + UNBIS/geo tagging
│   │   ├── dashboard_data_pipeline.py # Scoring pipeline
│   │   └── data_modules/              # Classification dictionaries
│   └── un_report_api/
│       └── app/
│           ├── main.py                # FastAPI application
│           ├── required_csvs/         # CSV fallback data for API
│           └── ...
├── db/
│   └── schema.sql                     # Turso/LibSQL DDL
├── docs/
│   ├── METHODOLOGY.md                 # Pillar computation formulas
│   └── SCHEMA.md                      # Database table reference
├── data/
│   └── reference/                     # Region mapping CSVs
├── cloudbuild.yaml                    # Google Cloud Build CI/CD
├── Dockerfile                         # Container image for API
├── requirements.txt                   # Python dependencies
└── .env.example                       # Environment variable template
```

## Data & Methodology

- Resolutions sourced from [UN Digital Library](https://digitallibrary.un.org)
- Subject classification uses the [UNBIS Thesaurus](https://metadata.un.org/thesaurus) (Main Category and Subcategory levels)
- Geographic tagging combines regex pattern matching with GPT-4o-mini verification
- Pillar scores are computed from voting records — see [docs/METHODOLOGY.md](docs/METHODOLOGY.md) for formulas
- Database schema fully documented in [docs/SCHEMA.md](docs/SCHEMA.md)

## Dependencies

Main dependencies:

- `openai` — GPT-4o-mini classification for scraper tagging
- `pandas` — Data manipulation and pipeline processing
- `numpy` — Numerical operations
- `scikit-learn` — Cosine similarity computation
- `selenium` — Web scraping automation
- `beautifulsoup4` — HTML parsing
- `webdriver-manager` — Chrome driver management
- `fastapi` — REST API framework
- `uvicorn` — ASGI server
- `pydantic` — Data validation
- `libsql-experimental` — Turso (LibSQL) Python client
- `python-dotenv` — Environment variable management
- `tqdm` — Progress indication
- `requests` — HTTP client
- `pycountry` — Country data utilities

## Citations

If you use this tool or data in your research, please cite:

```bibtex
@software{un_digital_library_scraper,
  author = {3DL},
  title = {UN Digital Library Scraper},
  year = {2025},
  url = {https://github.com/data-driven-decision-lab/un-digital-library-scraper}
}
```

## License

MIT License

Copyright (c) 2025 3DL

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
