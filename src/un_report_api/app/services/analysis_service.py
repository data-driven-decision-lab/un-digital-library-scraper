"""
Security Council Analysis Service using JSON data

This module provides analysis functionality for Security Council data,
reading from the comprehensive 'enhanced_veto_descriptions.json' file.
"""

import pandas as pd
import numpy as np
import json
import math
import os
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
from collections import defaultdict
# Removed veto_tagging import to eliminate external dependencies


def safe_float(value, default=0.0):
    """Convert value to a JSON-compliant float, handling NaN and Infinity."""
    if pd.isna(value) or value is None:
        return default
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return default
        return float(value)
    return default

logger = logging.getLogger(__name__)

class SecurityCouncilAnalysisService:
    """Analyzes comprehensive Security Council veto data from a JSON file."""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.data = []
        self._load_data()
    
    def _load_data(self):
        """Load data from the JSON file."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                logger.info(f"Successfully loaded {len(self.data)} records from {self.data_file}")
            else:
                logger.warning(f"Data file not found: {self.data_file}")
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error loading or parsing JSON data from {self.data_file}: {e}")
            self.data = []
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """Get comprehensive Security Council veto analysis."""
        if not self.data:
            return {"error": "No data available or failed to load."}
        
        total_resolutions = len(self.data)
        total_vetoes = sum(item.get('total_vetoes', 0) for item in self.data)
        
        # Country veto statistics
        country_stats = defaultdict(int)
        for item in self.data:
            for country in item.get('vetoing_countries', []):
                country_stats[country] += 1
        
        # Topic statistics
        topic_stats = defaultdict(int)
        for item in self.data:
            topic = item.get('canonical_topic', 'Unknown')
            topic_stats[topic] += 1
            
        # Year range analysis
        years = [item.get('year') for item in self.data if item.get('year')]
        analysis_period = f"{min(years)}-{max(years)}" if years else "N/A"
        
        # Power dynamics analysis
        power_dynamics = defaultdict(int)
        for item in self.data:
            dynamic = item.get('power_dynamic', 'UNKNOWN')
            power_dynamics[dynamic] += 1

        return {
            "summary": {
                "total_resolutions_analyzed": total_resolutions,
                "total_vetoes_cast": total_vetoes,
                "analysis_period": analysis_period,
                "unique_topics": len(topic_stats),
                "countries_with_vetoes": len(country_stats)
            },
            "vetoes_by_country": dict(country_stats),
            "resolutions_by_topic": dict(sorted(topic_stats.items(), key=lambda x: x[1], reverse=True)[:20]),  # Top 20 topics
            "power_dynamics_distribution": dict(power_dynamics),
            "data_source": self.data_file
        }
    
    def get_patterns_analysis(self) -> Dict[str, Any]:
        """Get veto patterns analysis."""
        if not self.data:
            return {"error": "No data available or failed to load."}

        # Veto frequency by country
        veto_patterns = defaultdict(int)
        for item in self.data:
            for country in item.get('vetoing_countries', []):
                veto_patterns[country] += 1
        
        # Power dynamics frequency
        power_dynamics = defaultdict(int)
        for item in self.data:
            dynamic = item.get('power_dynamic', 'UNKNOWN')
            power_dynamics[dynamic] += 1
        
        # Yearly veto trends
        yearly_vetoes = defaultdict(int)
        for item in self.data:
            year = item.get('year')
            if year:
                yearly_vetoes[year] += item.get('total_vetoes', 0)
        
        # Regional patterns
        regional_patterns = defaultdict(int)
        for item in self.data:
            region = item.get('region', 'Unknown')
            regional_patterns[region] += 1
            
        return {
            "patterns": [
                {
                    "pattern_type": "veto_frequency_by_country",
                    "description": "Total vetoes cast by each P5 member",
                    "data": dict(sorted(veto_patterns.items(), key=lambda x: x[1], reverse=True))
                },
                {
                    "pattern_type": "power_dynamics_frequency",
                    "description": "Frequency of observed power dynamics during veto events",
                    "data": dict(power_dynamics)
                },
                {
                    "pattern_type": "yearly_veto_trends",
                    "description": "Total vetoes cast per year",
                    "data": dict(sorted(yearly_vetoes.items()))
                },
                {
                    "pattern_type": "regional_focus",
                    "description": "Resolutions by affected region",
                    "data": dict(sorted(regional_patterns.items(), key=lambda x: x[1], reverse=True))
                }
            ],
            "data_source": self.data_file
        }
