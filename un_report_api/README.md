# UN Country Voting Report API

This FastAPI application provides an API endpoint to generate JSON reports on UN voting patterns for a specific country over a given time period.

## Setup

1.  **Create Directory Structure:**
    Ensure you have the main `un_report_api` directory and an `app` subdirectory within it.

2.  **Add Data Files:**
    -   Create a directory `un_report_api/app/dashboard_output/`.
    -   Copy your data CSV files (e.g., `annual_scores.csv`, `pairwise_similarity_yearly.csv`, `topic_votes_yearly.csv`) into this `dashboard_output` directory. The API's report generator script expects them here.

3.  **Install Dependencies:**
    Navigate to the `un_report_api` directory in your terminal and run:
    ```bash
    pip install -r requirements.txt
    ```

## Running the API

Navigate to the `un_report_api` directory (the one containing the `app` folder and `requirements.txt`).

Run the FastAPI application using Uvicorn:

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or, if you are inside the `un_report_api/app` directory:
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Breakdown of the command:
- `python -m uvicorn`: Runs Uvicorn as a module.
- `app.main:app`: Tells Uvicorn where to find the FastAPI application instance (`app`) located in `main.py` within the `app` package.
- `--reload`: Enables auto-reloading when code changes (useful for development).
- `--host 0.0.0.0`: Makes the server accessible on your network.
- `--port 8000`: Specifies the port to run on.

## API Usage

Once the server is running, you can access:

-   **API Documentation (Swagger UI):** `http://127.0.0.1:8000/docs`
-   **Alternative Documentation (ReDoc):** `http://127.0.0.1:8000/redoc`

### Endpoint: `GET /report/{country_iso}`

Generates a report for the specified country.

**Path Parameters:**
-   `country_iso` (string, 3 characters, uppercase): The ISO3 code of the country (e.g., `USA`).

**Query Parameters:**
-   `start_year` (integer): The start year for the report period. Must be between 1946 and 2024.
-   `end_year` (integer): The end year for the report period. Must be between 1946 and 2024. Must be greater than or equal to `start_year`. The period must span at least 3 years (`end_year` - `start_year` >= 2).

**Example Request:**
`http://127.0.0.1:8000/report/USA?start_year=2009&end_year=2013`

**Logging:**
Requests made to the API (ISO3 code, start date, end date) are logged to the console where Uvicorn is running. 