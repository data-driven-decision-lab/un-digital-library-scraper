#!/usr/bin/env python3
"""
Simple veto endpoint that returns enhanced veto data
"""

import pandas as pd
import json
import os
from typing import Dict, Any

def safe_str(value, default=''):
    """Safely convert value to string, handling NaN."""
    if pd.isna(value):
        return default
    return str(value)

def safe_int(value, default=0):
    """Safely convert value to int, handling NaN."""
    if pd.isna(value):
        return default
    try:
        return int(value)
    except:
        return default

def parse_json_field(field_value, default=None):
    """Parse JSON field safely."""
    if pd.isna(field_value) or not field_value or field_value == 'nan':
        return default if default is not None else []
    
    try:
        if isinstance(field_value, str):
            return json.loads(field_value)
        return field_value
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else []

def get_enhanced_veto_analysis() -> Dict[str, Any]:
    """Get enhanced veto analysis with comprehensive data."""
    
    # Load the enhanced data
    data_path = os.path.join(os.path.dirname(__file__), 'sc_data', 'fully_enhanced_veto_data.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Enhanced veto data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Create individual veto records
    individual_vetoes = []
    
    for _, row in df.iterrows():
        # Parse geographic and topic tags
        geographic_tags = parse_json_field(row.get('geographic_tags', '[]'), [])
        topic_tags = parse_json_field(row.get('topic_tags', '[]'), [])
        primary_countries = parse_json_field(row.get('primary_countries', '[]'), [])
        
        # Create P5 votes structure
        p5_votes = {
            "US": safe_int(row.get('us_veto', 0)),
            "RU": safe_int(row.get('russia_veto', 0)),
            "CN": safe_int(row.get('china_veto', 0)),
            "FR": safe_int(row.get('france_veto', 0)),
            "UK": safe_int(row.get('uk_veto', 0))
        }
        
        # Determine primary opposer
        primary_opposer = "Unknown"
        for country, votes in p5_votes.items():
            if votes > 0:
                country_map = {"US": "United States", "RU": "Russia", "CN": "China", "FR": "France", "UK": "United Kingdom"}
                primary_opposer = country_map.get(country, country)
                break
        
        record = {
            "id": safe_int(row.get('id')),
            "canonical_topic": safe_str(row.get('canonical_topic', '')),
            "full_resolution_name": safe_str(row.get('full_resolution_name', '')),
            "description": safe_str(row.get('enhanced_description', row.get('description', ''))),
            "historical_context": safe_str(row.get('enhanced_historical_context', '')),
            "resolution_type": safe_str(row.get('conflict_type', 'other')),
            "primary_countries": primary_countries,
            "confidence_score": 95,
            "year": safe_int(row.get('year', 0)),
            "draft_resolution": safe_str(row.get('draft_res_num', '')),
            "record_url": safe_str(row.get('url_for_record', '')),
            "resolution_url": safe_str(row.get('url_for_res', '')),
            "p5_votes": p5_votes,
            "total_vetoes": safe_int(row.get('total_vetoes', 0)),
            "primary_opposer": primary_opposer,
            "tags": {
                "geographic": geographic_tags,
                "topics": topic_tags
            }
        }
        individual_vetoes.append(record)
    
    # Sort by year (most recent first)
    individual_vetoes.sort(key=lambda x: x['year'], reverse=True)
    
    # Create summary
    summary = {
        'total_veto_occurrences': len(individual_vetoes),
        'total_topics': len(set(v['canonical_topic'] for v in individual_vetoes)),
        'year_range': f"{min(v['year'] for v in individual_vetoes)}-{max(v['year'] for v in individual_vetoes)}" if individual_vetoes else "N/A",
        'most_active_year': str(max(v['year'] for v in individual_vetoes)) if individual_vetoes else "N/A",
        'veto_counts': {
            'US': sum(v['p5_votes']['US'] for v in individual_vetoes),
            'RU': sum(v['p5_votes']['RU'] for v in individual_vetoes),
            'CN': sum(v['p5_votes']['CN'] for v in individual_vetoes),
            'FR': sum(v['p5_votes']['FR'] for v in individual_vetoes),
            'UK': sum(v['p5_votes']['UK'] for v in individual_vetoes)
        }
    }
    
    return {
        "summary": summary,
        "individual_vetoes": individual_vetoes
    }
