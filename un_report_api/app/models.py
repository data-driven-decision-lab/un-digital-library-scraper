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

class CountryAverageScoresPeriod(BaseModel):
    country_avg_pillar_1_score: Optional[float] = Field(default=None, example=90.5, description="Average Pillar 1 score for the selected country over the period.")
    country_avg_pillar_2_score: Optional[float] = Field(default=None, example=92.2, description="Average Pillar 2 score for the selected country over the period.")
    country_avg_pillar_3_score: Optional[float] = Field(default=None, example=85.8, description="Average Pillar 3 score for the selected country over the period.")
    country_avg_total_index_average: Optional[float] = Field(default=None, example=89.5, description="Average total index score for the selected country over the period.")
    # New fields for average ranks
    country_avg_pillar_1_rank: Optional[float] = Field(default=None, example=10.5, description="Average Pillar 1 rank for the selected country over the period.")
    country_avg_pillar_2_rank: Optional[float] = Field(default=None, example=12.0, description="Average Pillar 2 rank for the selected country over the period.")
    country_avg_pillar_3_rank: Optional[float] = Field(default=None, example=15.2, description="Average Pillar 3 rank for the selected country over the period.")
    country_avg_overall_rank: Optional[float] = Field(default=None, example=11.8, description="Average overall rank for the selected country over the period.")

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

# Define MostAlignedP5Member before it's used in ReportResponse
class MostAlignedP5Member(BaseModel):
    p5_country_iso3: Optional[str] = Field(default=None, example="CHN", description="ISO3 code of the P5 member.")
    average_similarity_score_scaled: Optional[float] = Field(default=None, example=75.5, description="Scaled average similarity score (0-100) with that P5 member over the period.")
    note: Optional[str] = Field(default=None, description="Contextual note, e.g., if the reported country is itself a P5 member.")

class LeastAlignedP5Member(BaseModel):
    p5_country_iso3: Optional[str] = Field(default=None, example="RUS", description="ISO3 code of the P5 member.")
    average_similarity_score_scaled: Optional[float] = Field(default=None, example=22.1, description="Scaled average similarity score (0-100) with that P5 member over the period.")
    note: Optional[str] = Field(default=None, description="Contextual note.")

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

# --- NEW: Models for Regional Context ---
class RegionalPeerAlignmentItem(BaseModel):
    country_iso3: str = Field(..., example="CAN")
    country_name: str = Field(..., example="Canada")
    alignment_score: Optional[float] = Field(default=None, example=82.3)

class RegionalContext(BaseModel):
    un_region: Optional[str] = Field(default=None, example="Northern America")
    regional_peer_alignment: List[RegionalPeerAlignmentItem] = []
    average_regional_alignment_score: Optional[float] = Field(default=None, example=78.5, description="Average alignment score of the selected country with its regional peers for the end_year.")
# --- END NEW ---

# --- Main Response Model ---
class ReportResponse(BaseModel):
    report_metadata: ReportMetadata
    world_average_scores_period: Optional[WorldAverageScoresPeriod]
    country_average_scores_period: Optional[CountryAverageScoresPeriod]
    index_score_analysis: IndexScoreAnalysis
    voting_behavior_overall: VotingBehaviorOverall
    most_aligned_p5_member: Optional[MostAlignedP5Member] = None
    least_aligned_p5_member: Optional[LeastAlignedP5Member] = None
    regional_context: Optional[RegionalContext] = None
    scores_timeseries: List[ScoreTimeseriesItem]
    top_allies: List[AllyEnemyItem]
    top_enemies: List[AllyEnemyItem]
    top_supported_topics: List[TopSupportedTopicItem]
    top_opposed_topics: List[TopOpposedTopicItem]
    all_topic_voting: List[TopicVoteItem]

# --- Models for Yearly Rankings Endpoint ---

class RankingEntry(BaseModel):
    country_name: str = Field(..., description="Name of the country.")
    value: Optional[float] = Field(default=None, description="Pillar score or average pillar score.")
    rank: Optional[int] = Field(default=None, description="Rank for the pillar based on the score (higher score is better).")
    rank_change: Optional[int] = Field(default=None, description="Change in rank compared to the previous year (e.g., a positive value means improved rank, negative means worsened).")
    value_change: Optional[float] = Field(default=None, description="Absolute change in score compared to the previous year.")

class YearlyPillarRankings(BaseModel):
    year: int = Field(..., description="The year for which rankings are provided.")
    pillar_1_rankings: List[RankingEntry] = Field(..., description="Rankings for Pillar 1.")
    pillar_2_rankings: List[RankingEntry] = Field(..., description="Rankings for Pillar 2.")
    pillar_3_rankings: List[RankingEntry] = Field(..., description="Rankings for Pillar 3.")
    average_pillar_rankings: List[RankingEntry] = Field(..., description="Rankings based on the average of Pillar 1, 2, and 3 (or total_index_average).")

class YearlyRankingsResponse(BaseModel):
    data: YearlyPillarRankings
    message: Optional[str] = Field(default=None, description="Contextual message, e.g., about data availability for change calculations.")

    class Config:
        # Example for the entire response, adjust as needed 
        pass # Added pass to fix indentation 