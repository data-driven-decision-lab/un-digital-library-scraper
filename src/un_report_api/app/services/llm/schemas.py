"""
LLM Output Schemas for Security Council Veto Analysis

This module defines strict Pydantic models for structured LLM outputs,
ensuring type safety and validation for all AI-generated content.
"""

from pydantic import BaseModel, Field, validator, StrictStr, StrictInt, StrictBool
from typing import List, Optional, Literal
from datetime import datetime
import re


class VetoResolutionSummary(BaseModel):
    """
    Structured output for LLM-generated veto resolution summaries.
    
    This model ensures that the LLM provides consistent, validated
    descriptions of vetoed UN Security Council resolutions.
    """
    
    # Core identification
    resolution_title: StrictStr = Field(
        ...,
        description="Full official title of the resolution (truncated if needed)",
        min_length=10,
        max_length=1500  # Increased to handle complex historical documents
    )
    
    # Main description (neutral, descriptive)
    description: StrictStr = Field(
        ...,
        description="One-sentence neutral description of what the resolution addressed (max 150 chars)",
        min_length=20,
        max_length=150
    )
    
    # Specific details for membership applications
    applicant_country: Optional[StrictStr] = Field(
        None,
        description="Name of country applying for UN membership (if applicable)",
        min_length=2,
        max_length=100
    )
    
    # Context and significance
    historical_context: Optional[StrictStr] = Field(
        None,
        description="Brief historical context or significance (max 200 chars)",
        max_length=200
    )
    
    # Resolution category
    resolution_type: Literal[
        "membership_application",
        "territorial_dispute", 
        "peacekeeping_operation",
        "sanctions_regime",
        "humanitarian_intervention",
        "arms_control",
        "human_rights",
        "other"
    ] = Field(
        ...,
        description="Category of the resolution"
    )
    
    # Confidence and quality indicators
    confidence_score: StrictInt = Field(
        ...,
        description="LLM confidence in the accuracy of the description (1-100)",
        ge=1,
        le=100
    )
    
    # Validation flags
    contains_country_name: StrictBool = Field(
        ...,
        description="Whether the description contains specific country/territory names"
    )
    
    is_descriptive_only: StrictBool = Field(
        ...,
        description="Whether the description avoids mentioning opposition or vetoes"
    )
    
    @validator('description')
    def validate_description_neutrality(cls, v):
        """Ensure description is neutral and doesn't mention opposition."""
        # Removed 'against' as it can be neutral in contexts like "complaint against"
        forbidden_words = [
            'vetoed', 'opposed', 'blocked', 'rejected',
            'objected', 'prevented', 'stopped', 'denied'
        ]
        v_lower = v.lower()
        for word in forbidden_words:
            if word in v_lower:
                raise ValueError(f"Description must be neutral and not mention '{word}'")
        return v
    
    @validator('description')
    def validate_description_format(cls, v):
        """Ensure description is a proper sentence."""
        if not v.endswith('.'):
            v += '.'
        if not v[0].isupper():
            v = v[0].upper() + v[1:]
        return v
    
    @validator('applicant_country')
    def validate_country_name(cls, v):
        """Validate country name format if provided."""
        if v is not None:
            # Remove common prefixes/suffixes that might be artifacts
            v = re.sub(r'^(the\s+|republic\s+of\s+)', '', v.lower()).title()
            v = re.sub(r'\s+(ssr|republic)$', '', v, flags=re.IGNORECASE)
        return v


class BatchVetoSummaries(BaseModel):
    """
    Container for multiple veto resolution summaries.
    Used for batch processing of multiple resolutions.
    """
    
    summaries: List[VetoResolutionSummary] = Field(
        ...,
        description="List of veto resolution summaries",
        min_items=1,
        max_items=50
    )
    
    processing_metadata: dict = Field(
        default_factory=dict,
        description="Metadata about the batch processing"
    )
    
    @validator('summaries')
    def validate_unique_resolutions(cls, v):
        """Ensure no duplicate resolution titles in batch."""
        titles = [summary.resolution_title for summary in v]
        if len(titles) != len(set(titles)):
            raise ValueError("Duplicate resolution titles found in batch")
        return v


class LLMProcessingResult(BaseModel):
    """
    Wrapper for LLM processing results with metadata.
    """
    
    success: StrictBool = Field(..., description="Whether processing was successful")
    
    result: Optional[VetoResolutionSummary] = Field(
        None,
        description="The processed veto summary (if successful)"
    )
    
    error_message: Optional[StrictStr] = Field(
        None,
        description="Error message (if processing failed)"
    )
    
    retry_count: StrictInt = Field(
        0,
        description="Number of retries attempted",
        ge=0,
        le=5
    )
    
    processing_time_ms: StrictInt = Field(
        ...,
        description="Processing time in milliseconds",
        ge=0
    )
    
    correlation_id: StrictStr = Field(
        ...,
        description="Unique identifier for tracking this processing request"
    )
    
    model_used: StrictStr = Field(
        ...,
        description="LLM model used for processing"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the processing was completed"
    )


class EnhancedVetoRecord(BaseModel):
    """
    Enhanced veto record with structured LLM-generated content.
    This replaces the simple dictionary structure with validated types.
    """
    
    # Core identification
    id: StrictInt = Field(..., description="Unique record identifier", ge=1)
    
    # Resolution information
    canonical_topic: StrictStr = Field(..., description="Canonical topic label")
    full_resolution_name: StrictStr = Field(..., description="Full official resolution name")
    
    # LLM-generated structured content
    resolution_summary: VetoResolutionSummary = Field(
        ...,
        description="Structured LLM-generated summary"
    )
    
    # Temporal information
    year: StrictInt = Field(..., description="Year of the veto", ge=1946, le=2030)
    
    # Geographic information
    country: StrictStr = Field(..., description="ISO country code")
    region: StrictStr = Field(..., description="UN geoscheme region")
    
    # Voting information
    p5_votes: dict = Field(..., description="P5 voting patterns")
    vetoing_countries: List[StrictStr] = Field(..., description="Countries that vetoed")
    total_vetoes: StrictInt = Field(..., description="Total number of vetoes", ge=0)
    
    # Analysis fields
    primary_opposer: StrictStr = Field(..., description="Primary opposing country")
    controversy_score: float = Field(..., description="Controversy score", ge=0.0, le=1.0)
    power_dynamic: StrictStr = Field(..., description="Power dynamic classification")
    dominant_bloc: StrictStr = Field(..., description="Dominant bloc classification")
    overton_position: StrictStr = Field(..., description="Overton window position")
    
    @validator('vetoing_countries')
    def validate_vetoing_countries(cls, v):
        """Ensure vetoing countries list is not empty."""
        if not v:
            raise ValueError("At least one vetoing country must be specified")
        return v
    
    @validator('p5_votes')
    def validate_p5_votes_structure(cls, v):
        """Ensure P5 votes has correct structure."""
        required_keys = {'US', 'RU', 'CN', 'FR', 'UK'}
        if not all(key in v for key in required_keys):
            raise ValueError(f"P5 votes must contain all keys: {required_keys}")
        return v

