"""FastAPI application for UN Country Voting Report API with CORS enabled."""

import logging
import os
import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Path, Query, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Tuple
import re
#test
# Use absolute imports with proper path handling
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Note: Removed old security_council path references - now using sc_analysis folder

from report_generator import generate_report
from models import (
    ReportResponse,
    MIN_YEAR_CONSTRAINT,
    MAX_YEAR_CONSTRAINT,
    YearlyRankingsResponse,
    YearlyPillarRankings,
    SecurityCouncilTopicResponse,
    SecurityCouncilPolicyReport,
    AnnualScoresResponse,
    SecurityCouncilTopicItem,
    AnnualScoresItem
)

# Import Security Council services from new location
from services.analysis_service import SecurityCouncilAnalysisService
from services.vote_analysis_service import SecurityCouncilVoteAnalysisService
from services.data_loader import SecurityCouncilDataLoader
from ranking_generator import generate_yearly_rankings
from simple_veto_endpoint import get_enhanced_veto_analysis

# Note: Removed unused LLM comparison imports (TopicEntityResolver, PowerDynamicsAnalyzer)
# These were never used in any endpoints and are part of the research thesis, not the production API

# Security Council analysis imports
try:
    from supabase import create_client
    from un_classification_mapper import UNClassificationMapper, classify_title_with_un_mapper
except ImportError:
    # Fallback for development
    create_client = None
    UNClassificationMapper = None

import json
import pandas as pd
from typing import List, Optional

# --- FastAPI App Initialization ---
app = FastAPI(
    title="UN Country Voting Report API",
    description="Generates a JSON report for UN voting patterns of a specific country over a time period (Years: 1946-2024).",
    version="1.2.0" # Incremented version due to new endpoint and CORS update
)

# --- CORS Middleware Configuration (User Specified) ---
# Enable CORS for datadrivendecisionlab.com
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://datadrivendecisionlab.com"],  # allow specific origin
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
    ),
    recent_year_only: bool = Query(
        False,
        description="If true, calculates stats for only the most recent year period (2023-2024). When enabled, start_year and end_year are ignored and automatically set to 2023-2024."
    )
) -> Dict[str, int]:
    # If recent_year_only is enabled, override the year parameters
    if recent_year_only:
        return {"start_year": 2023, "end_year": 2024}
    
    if end_year < start_year:
        raise HTTPException(status_code=400, detail="End year cannot be before start year.")
    return {"start_year": start_year, "end_year": end_year}

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "UN Country Voting Report API. Access /docs for API documentation."}

# --- Health Check Endpoint ---
@app.get("/health", tags=["Health"])
async def health_check():
    """Check the health of the API."""
    api_logger.info("Health check endpoint called.")
    return {"status": "ok"}

# --- Classic UN Voting Analysis Endpoints ---

@app.get(
    "/report/{country_iso}",
    response_model=ReportResponse,
    tags=["Country Reports"],
    summary="Generate a country voting report",
    description="Provides a detailed report on a country's UN voting patterns for a specified period. Use recent_year_only=true to get stats for just the most recent year (2023-2024)."
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

    # Check if this was a recent year request by seeing if it's 2023-2024
    is_recent_year_request = start_year == 2023 and end_year == 2024
    recent_year_suffix = " (recent year only)" if is_recent_year_request else ""
    
    api_logger.info(f"Processing report generation for ISO: {country_iso}, Start: {start_year}, End: {end_year}{recent_year_suffix}")

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

# --- Yearly Rankings Endpoint ---
@app.get(
    "/rankings/{year}",
    response_model=YearlyRankingsResponse,
    tags=["Country Rankings"],
    summary="Get yearly pillar rankings for all countries",
    description="Provides rankings for Pillar 1, Pillar 2, Pillar 3, and Average Pillar score for a specified year, including changes from the previous year."
)
async def get_yearly_rankings_api(
    year: int = Path(
        ..., 
        ge=MIN_YEAR_CONSTRAINT,
        le=MAX_YEAR_CONSTRAINT, 
        description=f"Year for the rankings (inclusive, {MIN_YEAR_CONSTRAINT}-{MAX_YEAR_CONSTRAINT})."
    )
):
    api_logger.info(f"Received request for yearly rankings for year: {year}")
    try:
        rankings_data, message = generate_yearly_rankings(year)
        
        # Construct the Pydantic model for the response
        # The generate_yearly_rankings now returns a dict that should match YearlyPillarRankings structure
        # plus a message string. We need to wrap rankings_data in YearlyPillarRankings if it's not already.
        # Based on ranking_generator, rankings_data is already the correct dict for YearlyPillarRankings.
        
        response_data = YearlyPillarRankings(**rankings_data) # rankings_data should be the dict for the model

        api_logger.info(f"Successfully generated yearly rankings for {year}.")
        return YearlyRankingsResponse(data=response_data, message=message)

    except FileNotFoundError as e:
        api_logger.error(f"Data file not found for yearly rankings: {e.filename if hasattr(e, 'filename') else e}", exc_info=True)
        raise HTTPException(
            status_code=503, 
            detail=f"A required data file ('{e.filename if hasattr(e, 'filename') else 'annual_scores.csv'}') for ranking generation is missing."
        )
    except ValueError as e:
        api_logger.warning(f"Validation or data error for yearly rankings ({year}): {str(e)}")
        # This could be due to missing columns or no data for the year after file load
        # If generate_yearly_rankings returns specific structured error for no data, that might be handled differently
        # For now, treating as a 404 or 400 depending on the nature of ValueError
        if "No data available for the year" in str(e) or "No data found for the year" in str(e):
             raise HTTPException(status_code=404, detail=str(e))
        elif "missing required columns" in str(e):
            raise HTTPException(status_code=500, detail=f"Internal configuration error: {str(e)}") # Data is malformed
        else:
            raise HTTPException(status_code=400, detail=str(e)) # Other value errors like invalid year logic if not caught by Path
    except Exception as e:
        api_logger.error(f"Unexpected error generating yearly rankings for {year}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"An internal server error occurred while generating yearly rankings. Error type: {type(e).__name__}"
        )

# --- Security Council Analysis Endpoints ---






# --- Security Council Analysis Endpoints ---


@app.get(
    "/sc/veto_analysis",
    tags=["Security Council Analysis"],
    summary="Get Security Council veto analysis",
    description="Returns comprehensive analysis of Security Council veto patterns, power dynamics, and behavioral trends. Focuses on veto occurrences and P5 behavior patterns."
)
async def get_security_council_veto_analysis():
    """Get Security Council veto analysis with canonical labels and deterministic statistics."""
    api_logger.info("Processing Security Council veto analysis request")

    try:
        # Initialize data loader and analysis service
        data_loader = SecurityCouncilDataLoader()
        
        # Try to get the fully enhanced data first, then fallback to available data
        fully_enhanced_data_path = os.path.join(data_loader.base_path, 'fully_enhanced_veto_data.csv')
        if data_loader.file_exists(fully_enhanced_data_path):
            data_file = fully_enhanced_data_path
        else:
            data_file = data_loader.get_available_data_file()
        
        if not data_file:
            raise HTTPException(
                status_code=503,
                detail="Security Council analysis data is not available."
            )
        
        # Generate enhanced analysis using the simple endpoint helper
        result = get_enhanced_veto_analysis()
        
        api_logger.info("Successfully generated Security Council analysis")
        return result

    except Exception as e:
        api_logger.error(f"Error in Security Council analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Security Council analysis not available: {str(e)}"
        )

@app.get(
    "/sc/vote_analysis",
    tags=["Security Council Analysis"],
    summary="Get Security Council vote analysis",
    description="Provides comprehensive analysis of Security Council voting patterns, consensus metrics, and behavioral trends. Focuses on complete voting behavior rather than just vetoes."
)
async def get_security_council_vote_analysis():
    """Get comprehensive Security Council vote analysis covering all voting patterns."""
    api_logger.info("Processing Security Council vote analysis request")

    try:
        # Initialize data loader and vote analysis service
        data_loader = SecurityCouncilDataLoader()
        vote_data_file = data_loader.get_vote_analysis_data_path()
        
        if not data_loader.file_exists(vote_data_file):
            raise HTTPException(
                status_code=503,
                detail="Security Council vote analysis data is not available."
            )
        
        # Initialize vote analysis service with the data file
        vote_analysis_service = SecurityCouncilVoteAnalysisService(vote_data_file)
        
        # Generate comprehensive vote analysis
        result = vote_analysis_service.get_comprehensive_vote_analysis()
        
        api_logger.info("Successfully generated Security Council vote analysis")
        return result

    except Exception as e:
        api_logger.error(f"Error in Security Council vote analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Security Council vote analysis not available: {str(e)}"
        )

# Run with Uvicorn example (CLI):
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
