"""
Security Council Vote Analysis Service

This module provides comprehensive analysis of Security Council voting patterns,
focusing on complete voting behavior rather than just vetoes.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import os
from itertools import combinations
from collections import defaultdict
from .data_loader import SecurityCouncilDataLoader

# Set up logging
logger = logging.getLogger(__name__)
logger.info("--- vote_analysis_service.py (v2) loaded ---")


class SecurityCouncilVoteAnalysisService:
    """Service class for Security Council vote analysis operations."""
    
    def __init__(self, data_path: Optional[str] = None, cache_path: Optional[str] = None):
        """
        Initialize the vote analysis service.

        Args:
            data_path: Path to the UN votes dataset CSV file
            cache_path: Path to the pre-calculated analysis CSV file
        """
        loader = SecurityCouncilDataLoader()
        default_data_path = loader.get_vote_analysis_data_path()
        default_cache_path = os.path.join(loader.base_path, 'p5_vote_analysis_voting_v2.csv')

        self.data_path = data_path or default_data_path
        self.cache_path = cache_path or default_cache_path
        self._sc_data = None
        self._p5_countries = ['USA', 'CHN', 'RUS', 'GBR', 'FRA']

    def load_data(self) -> pd.DataFrame:
        """Load and filter Security Council data from the UN votes dataset."""
        if self._sc_data is not None:
            return self._sc_data

        if not self.data_path or not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Vote analysis data file not found: {self.data_path}")

        # Load the complete dataset
        df = pd.read_csv(self.data_path, low_memory=False)
        logger.info(f"Loaded UN votes dataset with {len(df)} records")

        # Strip whitespace from column headers to prevent KeyErrors
        df.columns = df.columns.str.strip()

        # Filter for Security Council votes only
        if 'Resolution' not in df.columns:
            raise ValueError("The 'Resolution' column is missing from the CSV file.")

        df['Resolution'] = df['Resolution'].astype(str)
        self._sc_data = df[df['Resolution'].str.startswith('S/', na=False)].copy()

        if self._sc_data.empty:
            raise ValueError("No Security Council resolutions found in the provided CSV file. Please check the data source.")

        logger.info(f"Filtered to {len(self._sc_data)} Security Council resolutions")

        # Clean and prepare data
        self._sc_data['Date'] = pd.to_datetime(self._sc_data['Date'], errors='coerce')
        self._sc_data['Year'] = self._sc_data['Date'].dt.year

        return self._sc_data
    
    def get_comprehensive_vote_analysis(self) -> Dict[str, Any]:
        """
        Generate a simplified P5 vote analysis.
        
        Returns:
            Dictionary containing P5 vote analysis results
        """
        sc_data = self.load_data() # Load the data once and store it

        if os.path.exists(self.cache_path):
            logger.info(f"Loading analysis from cache: {self.cache_path}")
            yearly_analysis = self._load_analysis_from_csv()
        else:
            logger.info("Cache not found. Generating analysis...")
            yearly_analysis = self._calculate_yearly_p5_analysis(sc_data)
            self._save_analysis_to_csv(yearly_analysis)
        
        result = {
            "p5_yearly_behavior": yearly_analysis,
            "analysis_metadata": {
                "analysis_type": "P5 Security Council Voting Analysis",
                "data_source": "UN votes dataset filtered for Security Council resolutions",
                "focus": "Yearly P5 voting patterns and pairwise voting similarity.",
                "methodology": "Statistical analysis of P5 voting records (YES/NO/ABSTAIN) on Security Council resolutions.",
                "temporal_coverage": f"{sc_data['Year'].min()}-{sc_data['Year'].max()}",
                "total_resolutions": len(sc_data)
            }
        }
        # Debug: log a sample of keys to verify structure at runtime
        if isinstance(yearly_analysis, list) and yearly_analysis:
            logger.info(f"Sample yearly_analysis keys: {list(yearly_analysis[0].keys())}")
        
        logger.info("Successfully generated P5 Security Council voting analysis")
        return result

    def _load_analysis_from_csv(self) -> List[Dict[str, Any]]:
        """Loads yearly P5 analysis from a CSV file."""
        df = pd.read_csv(self.cache_path)
        return df.to_dict('records')

    def _save_analysis_to_csv(self, yearly_analysis: List[Dict[str, Any]]):
        """Saves the yearly P5 analysis to a CSV file."""
        df = pd.DataFrame(yearly_analysis)
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        df.to_csv(self.cache_path, index=False)
        logger.info(f"Analysis saved to cache: {self.cache_path}")

    def _calculate_yearly_p5_analysis(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyzes P5 voting patterns on a yearly basis."""
        yearly_results = []

        for year in sorted(data['Year'].unique()):
            if pd.isna(year):
                continue

            year_data = data[data['Year'] == year]

            year_record = {'year': int(year)}

            # Vote counts for P5 members
            for country in self._p5_countries:
                if country in year_data.columns:
                    votes = year_data[country].value_counts()
                    year_record[f'yes_{country}'] = int(votes.get('YES', 0))
                    year_record[f'no_{country}'] = int(votes.get('NO', 0))
                    year_record[f'abstain_{country}'] = int(votes.get('ABSTAIN', 0))
                else:
                    year_record[f'yes_{country}'] = 0
                    year_record[f'no_{country}'] = 0
                    year_record[f'abstain_{country}'] = 0

            # Pairwise similarity based on voting patterns
            for country1, country2 in combinations(self._p5_countries, 2):
                pair_key = f"{country1}-{country2}"
                if country1 in year_data.columns and country2 in year_data.columns:
                    votes = year_data[[country1, country2]].dropna()
                    if not votes.empty:
                        agreements = (votes[country1] == votes[country2]).sum()
                        total_votes = len(votes)
                        score = agreements / total_votes if total_votes > 0 else 0
                        year_record[f'similarity_{pair_key}'] = round(score, 4)
                    else:
                        year_record[f'similarity_{pair_key}'] = 0
                else:
                    year_record[f'similarity_{pair_key}'] = 0

            # Consensus Level - based on YES votes proportion
            if not year_data.empty and 'YES COUNT' in year_data and 'TOTAL VOTES' in year_data:
                valid_votes = year_data['TOTAL VOTES'] > 0
                if valid_votes.any():
                    consensus = (year_data.loc[valid_votes, 'YES COUNT'] / year_data.loc[valid_votes, 'TOTAL VOTES']).mean()
                    year_record['consensus_level'] = round(consensus, 4)
                else:
                    year_record['consensus_level'] = 0
            else:
                year_record['consensus_level'] = 0

            yearly_results.append(year_record)

        return yearly_results
