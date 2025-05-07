"""FastAPI application for UN Country Voting Report API with CORS enabled."""

import logging
from fastapi import FastAPI, Path, Query, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

# Relative imports for when running as a module/package
from .report_generator import generate_report
from .models import ReportResponse, MIN_YEAR_CONSTRAINT, MAX_YEAR_CONSTRAINT

# --- FastAPI App Initialization ---
app = FastAPI(
    title="UN Country Voting Report API",
    description="Generates a JSON report for UN voting patterns of a specific country over a time period (Years: 1946-2024).",
    version="1.2.0" # Incremented version due to new endpoint and CORS update
)

# --- CORS Middleware Configuration (User Specified) ---
# Enable CORS for all origins, methods, headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # allow any origin
    allow_credentials=True,     # allow cookies, Authorization headers, etc.
    allow_methods=["*"],        # allow all HTTP methods (GET, POST, PUT, DELETE, …)
    allow_headers=["*"],        # allow all headers
)

# --- Logging Configuration ---
api_logger = logging.getLogger("un_report_api")
api_logger.setLevel(logging.INFO)
if not api_logger.hasHandlers():
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    api_logger.addHandler(stream_handler)

# --- Request Logging Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    api_logger.info(f"Incoming request: {request.method} {request.url.path} Query: {request.query_params}")
    response = await call_next(request)
    api_logger.info(f"Response status: {response.status_code}")
    return response

# --- Dependency for Year Validation ---
async def validate_year_params(
    start_year: int = Query(
        ..., ge=MIN_YEAR_CONSTRAINT, le=MAX_YEAR_CONSTRAINT,
        description=f"Start year of the period (inclusive, {MIN_YEAR_CONSTRAINT}-{MAX_YEAR_CONSTRAINT})."
    ),
    end_year: int = Query(
        ..., ge=MIN_YEAR_CONSTRAINT, le=MAX_YEAR_CONSTRAINT,
        description=f"End year of the period (inclusive, {MIN_YEAR_CONSTRAINT}-{MAX_YEAR_CONSTRAINT})."
    )
) -> Dict[str, int]:
    if end_year < start_year:
        raise HTTPException(status_code=400, detail="End year cannot be before start year.")
    if (end_year - start_year) < 2:
        raise HTTPException(
            status_code=400,
            detail="The period must span at least 3 years (e.g., 2000-2002 means end_year - start_year >= 2)."
        )
    return {"start_year": start_year, "end_year": end_year}

# --- API Endpoint ---
@app.get(
    "/report/{country_iso}",
    response_model=ReportResponse,
    tags=["Country Reports"],
    summary="Generate a country voting report",
    description="Provides a detailed report on a country's UN voting patterns for a specified period."
)
async def get_country_report_api(
    country_iso: str = Path(
        ..., min_length=3, max_length=3, pattern="^[A-Z]{3}$",
        description="3-letter uppercase ISO code of the country (e.g., USA).",
        example="USA"
    ),
    year_params: Dict[str, int] = Depends(validate_year_params)
):
    start_year = year_params["start_year"]
    end_year = year_params["end_year"]

    api_logger.info(f"Processing report generation for ISO: {country_iso}, Start: {start_year}, End: {end_year}")

    try:
        report_data = generate_report(
            country_iso=country_iso,
            start_year=start_year,
            end_year=end_year
        )
        api_logger.info(f"Successfully generated report for {country_iso}, {start_year}-{end_year}")
        return report_data

    except FileNotFoundError as e:
        api_logger.error(f"Prerequisite data file not found: {e.filename}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"A required data file ('{e.filename}') for report generation is missing on the server."
        )
    except ValueError as e:
        api_logger.warning(
            f"Data or validation error during report generation for {country_iso} ({start_year}-{end_year}): {str(e)}"
        )
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        api_logger.error(
            f"An unexpected error occurred while generating report for {country_iso} ({start_year}-{end_year}): {str(e)}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"An internal server error occurred. Please contact support. Error type: {type(e).__name__}"
        )

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "UN Country Voting Report API. Access /docs for API documentation."}

# --- Health Check Endpoint (New) ---
@app.get("/health", tags=["Health"])
async def health_check():
    """Check the health of the API."""
    api_logger.info("Health check endpoint called.")
    return {"status": "ok"}

# Run with Uvicorn example (CLI):
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
