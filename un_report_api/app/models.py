"""Pydantic models for API request and response validation."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict

# --- Constants for Validation (as per user request) ---
MIN_YEAR_CONSTRAINT = 1946
MAX_YEAR_CONSTRAINT = 2024 # Fixed max year

# --- Nested Models for the Report Structure ---
class ReportMetadata(BaseModel):
    country_iso3: str = Field(..., example="USA", description="3-letter ISO code of the country.")
    country_name: str = Field(..., example="United States of America", description="Full name of the country.")
    start_year: int = Field(..., example=2009, description="Start year of the reporting period.")
    end_year: int = Field(..., example=2013, description="End year of the reporting period.")

class WorldAverageScoresPeriod(BaseModel):
    world_avg_pillar_1_score: Optional[float] = Field(default=None, example=50.5)
    world_avg_pillar_2_score: Optional[float] = Field(default=None, example=60.2)
    world_avg_pillar_3_score: Optional[float] = Field(default=None, example=45.8)
    world_avg_total_index_average: Optional[float] = Field(default=None, example=52.1)

class IndexScoreAnalysis(BaseModel):
    end_year_score: Optional[float] = Field(default=None, example=85.5)
    start_year_score: Optional[float] = Field(default=None, example=80.1)
    percentage_change: Optional[float] = Field(default=None, example=6.74)
    start_year_rank: Optional[int] = Field(default=None, example=10)
    end_year_rank: Optional[int] = Field(default=None, example=8)

class VotingBehaviorOverall(BaseModel):
    total_votes: Optional[int] = Field(default=None, example=1000) # Made optional as sum could be 0
    yes_votes: Optional[int] = Field(default=None, example=700)
    no_votes: Optional[int] = Field(default=None, example=200)
    abstain_votes: Optional[int] = Field(default=None, example=100)
    yes_percentage: Optional[float] = Field(default=None, example=70.0)
    no_percentage: Optional[float] = Field(default=None, example=20.0)
    abstain_percentage: Optional[float] = Field(default=None, example=10.0)
    # World percentages for the period
    world_yes_percentage: Optional[float] = Field(default=None, example=65.0, description="World average YES vote percentage for the period.")
    world_no_percentage: Optional[float] = Field(default=None, example=25.0, description="World average NO vote percentage for the period.")
    world_abstain_percentage: Optional[float] = Field(default=None, example=10.0, description="World average ABSTAIN vote percentage for the period.")
    # Difference vs world average
    yes_vs_world_avg: Optional[float] = Field(default=None, example=5.0)
    no_vs_world_avg: Optional[float] = Field(default=None, example=-2.5)
    abstain_vs_world_avg: Optional[float] = Field(default=None, example=-2.5)

class ScoreTimeseriesItem(BaseModel):
    year: int = Field(..., example=2010)
    pillar_1_score: Optional[float] = None
    pillar_2_score: Optional[float] = None
    pillar_3_score: Optional[float] = None
    total_index_average: Optional[float] = None
    pillar_1_rank: Optional[int] = None
    pillar_2_rank: Optional[int] = None
    pillar_3_rank: Optional[int] = None
    overall_rank: Optional[int] = None
    world_avg_pillar_1_score: Optional[float] = None
    world_avg_pillar_2_score: Optional[float] = None
    world_avg_pillar_3_score: Optional[float] = None
    world_avg_total_index_average: Optional[float] = None

class AllyEnemyItem(BaseModel):
    country: str = Field(..., example="CAN")
    average_similarity_score_scaled: Optional[float] = Field(default=None, example=95.5)

class TopicVoteItem(BaseModel):
    topictag: str = Field(..., example="Topic A")
    support_percentage: Optional[float] = None
    opposition_percentage: Optional[float] = None
    abstain_percentage: Optional[float] = None
    world_support_percentage: Optional[float] = None
    world_opposition_percentage: Optional[float] = None
    world_abstain_percentage: Optional[float] = None
    support_vs_world_avg: Optional[float] = None
    opposition_vs_world_avg: Optional[float] = None
    abstain_vs_world_avg: Optional[float] = None
    total_yes: Optional[int] = None
    total_no: Optional[int] = None
    total_abstain: Optional[int] = None
    total_votes: Optional[int] = None

class TopSupportedTopicItem(BaseModel):
    topictag: str = Field(..., example="Topic B")
    support_percentage: Optional[float] = None
    world_support_percentage: Optional[float] = None
    support_vs_world_avg: Optional[float] = None
    total_yes: Optional[int] = None
    total_votes: Optional[int] = None

class TopOpposedTopicItem(BaseModel):
    topictag: str = Field(..., example="Topic C")
    opposition_percentage: Optional[float] = None
    world_opposition_percentage: Optional[float] = None
    opposition_vs_world_avg: Optional[float] = None
    total_no: Optional[int] = None
    total_votes: Optional[int] = None

# --- Main Response Model ---
class ReportResponse(BaseModel):
    report_metadata: ReportMetadata
    world_average_scores_period: Optional[WorldAverageScoresPeriod]
    index_score_analysis: IndexScoreAnalysis
    voting_behavior_overall: VotingBehaviorOverall
    scores_timeseries: List[ScoreTimeseriesItem]
    top_allies: List[AllyEnemyItem]
    top_enemies: List[AllyEnemyItem]
    top_supported_topics: List[TopSupportedTopicItem]
    top_opposed_topics: List[TopOpposedTopicItem]
    all_topic_voting: List[TopicVoteItem] 