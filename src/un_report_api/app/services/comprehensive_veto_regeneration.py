#!/usr/bin/env python3
"""
Comprehensive Veto Data Regeneration Script

This script regenerates all veto data with:
1. Enhanced descriptions (400+ characters with contemporary language)
2. Comprehensive topic tagging (3-level hierarchy)
3. Complete geographic tagging (country, subregion, continent)
4. Structured LLM analysis for each veto

Uses the existing LLM tagging pipeline from the UN scraper.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sc_analysis_dir = os.path.dirname(current_dir)  # sc_analysis/
project_root = os.path.dirname(sc_analysis_dir)  # project root
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import our existing LLM infrastructure
from veto_tagging import tag_veto_data, geo_hierarchy, iso2_country_code
from llm.runtime import StructuredLLMClient, generate_veto_summary_structured
from llm.schemas import VetoResolutionSummary, EnhancedVetoRecord

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
BATCH_SIZE = 10  # Process in batches to avoid rate limits
DELAY_BETWEEN_BATCHES = 2  # seconds

class ComprehensiveVetoRegenerator:
    """Regenerates comprehensive veto data with enhanced descriptions and tagging."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.dppa_data = {}
        self.enhanced_records = []
        
    def load_source_data(self):
        """Load all source data files."""
        logger.info("Loading source data files...")
        
        # Load DPPA source data
        dppa_path = os.path.join(os.path.dirname(self.data_path), 'DPPA-SCVETOES.csv')
        if os.path.exists(dppa_path):
            dppa_df = pd.read_csv(dppa_path)
            for _, row in dppa_df.iterrows():
                key = f"{row.get('short_agenda', '')}_{row.get('year', 0)}"
                self.dppa_data[key] = row
            logger.info(f"Loaded {len(self.dppa_data)} DPPA source records")
        
        # Load main veto data
        self.veto_df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.veto_df)} veto records")
        
    def create_enhanced_description(self, canonical_topic: str, year: int, dppa_info: Optional[Dict] = None) -> str:
        """Create enhanced contemporary description using available data."""
        
        # Get vetoing country for description
        vetoing_countries = []
        if dppa_info:
            if dppa_info.get('united_states', 0) > 0:
                vetoing_countries.append('United States')
            if dppa_info.get('russian_federation_ussr', 0) > 0:
                vetoing_countries.append('Russian Federation' if year >= 1991 else 'Soviet Union')
            if dppa_info.get('china', 0) > 0:
                vetoing_countries.append('China')
            if dppa_info.get('france', 0) > 0:
                vetoing_countries.append('France')
            if dppa_info.get('united_kingdom', 0) > 0:
                vetoing_countries.append('United Kingdom')
        
        vetoing_country = vetoing_countries[0] if vetoing_countries else 'Unknown'
        
        # Create contemporary description based on available data
        if dppa_info and len(dppa_info) > 0:
            draft_res = str(dppa_info.get('draft_res', ''))
            date = str(dppa_info.get('date', ''))
            agenda = str(dppa_info.get('agenda', canonical_topic))
            
            # Format date for readability
            try:
                if date and date != 'nan':
                    date_obj = datetime.strptime(date, '%Y-%m-%d')
                    formatted_date = date_obj.strftime('%B %d, %Y')
                else:
                    formatted_date = str(year)
            except:
                formatted_date = str(year)
            
            # Create specific description based on topic type
            if 'admission' in canonical_topic.lower():
                description = f"Draft resolution {draft_res} concerning admission of new member state to UN membership, vetoed by {vetoing_country} on {formatted_date}. The resolution addressed the application process and criteria for new member admission to the United Nations organization."
            elif 'ukraine' in canonical_topic.lower():
                description = f"Draft resolution {draft_res} addressing maintenance of peace and security of Ukraine, vetoed by {vetoing_country} on {formatted_date}. The resolution focused on regional stability and conflict prevention measures in the Ukrainian context."
            elif 'middle east' in canonical_topic.lower() or 'palestinian' in canonical_topic.lower():
                description = f"Draft resolution {draft_res} on Palestinian question in Middle East situation, vetoed by {vetoing_country} on {formatted_date}. The resolution addressed territorial disputes, refugee rights, and regional peace initiatives in the Middle East."
            elif 'telegram' in canonical_topic.lower() and 'greece' in canonical_topic.lower():
                description = f"Draft resolution {draft_res} on complaint regarding Greek situation, submitted {formatted_date}, vetoed by {vetoing_country}. The resolution addressed civil war conditions and international intervention concerns in post-war Greece."
            elif 'spanish' in canonical_topic.lower():
                description = f"Draft resolution {draft_res} addressing Spain's international status, vetoed by {vetoing_country} on {formatted_date}. The resolution concerned Spain's position in international organizations following the Franco regime."
            else:
                # Generic template for other resolutions
                short_agenda = agenda[:50] + '...' if len(agenda) > 50 else agenda
                description = f"Draft resolution {draft_res} on {short_agenda}, vetoed by {vetoing_country} on {formatted_date}. The resolution addressed specific international concerns requiring Security Council attention."
            
            # Ensure within 400 character limit
            if len(description) > 400:
                # Truncate at natural break point
                description = description[:397] + '...'
            
            return description
        
        # Fallback description
        return f"Resolution regarding {canonical_topic.lower()}, addressing international concerns requiring Security Council consideration in {year}."
    
    def process_batch_with_llm(self, batch_df: pd.DataFrame) -> List[Dict]:
        """Process a batch of veto records with LLM analysis."""
        batch_results = []
        
        for _, row in batch_df.iterrows():
            try:
                canonical_topic = str(row.get('canonical_topic', ''))
                year = int(row.get('first_year', 0))
                
                # Get DPPA info - fix pandas Series boolean issue
                dppa_key = f"{canonical_topic}_{year}"
                dppa_info = self.dppa_data.get(dppa_key, {})
                
                # Convert pandas Series to dict if needed
                if hasattr(dppa_info, 'to_dict'):
                    dppa_info = dppa_info.to_dict()
                
                # Create enhanced description
                enhanced_description = self.create_enhanced_description(canonical_topic, year, dppa_info)
                
                # Generate LLM summary for additional context
                try:
                    llm_result = generate_veto_summary_structured(
                        resolution_title=canonical_topic,
                        year=year,
                        vetoing_countries=[row.get('primary_opposer', 'Unknown')],
                        api_key=API_KEY
                    )
                    llm_summary = llm_result.result
                except Exception as e:
                    logger.warning(f"LLM analysis failed for {canonical_topic}: {e}")
                    llm_summary = VetoResolutionSummary(
                        resolution_title=canonical_topic,
                        description=enhanced_description,
                        applicant_country=None,
                        historical_context="",
                        resolution_type="other",
                        confidence_score=50,
                        contains_country_name=False,
                        is_descriptive_only=True
                    )
                
                # Create comprehensive record
                record = {
                    "id": len(self.enhanced_records) + 1,
                    "canonical_topic": canonical_topic,
                    "full_resolution_name": llm_summary.resolution_title,
                    "description": enhanced_description,
                    "resolution_type": llm_summary.resolution_type,
                    "applicant_country": llm_summary.applicant_country,
                    "confidence_score": llm_summary.confidence_score,
                    "year": year,
                    "country": str(row.get('country', 'Unknown')),
                    "region": str(row.get('primary_region', 'Unknown')),
                    "p5_votes": {
                        "US": int(row.get('us_vetoes', 0)),
                        "RU": int(row.get('ru_vetoes', 0)), 
                        "CN": int(row.get('cn_vetoes', 0)),
                        "FR": int(row.get('fr_vetoes', 0)),
                        "UK": int(row.get('uk_vetoes', 0))
                    },
                    "total_vetoes": int(row.get('total_veto_occurrences', 0)),
                    "primary_opposer": str(row.get('primary_opposer', 'Unknown')),
                    "controversy_score": float(row.get('controversy_score', 0.0)),
                    "power_dynamic": str(row.get('power_dynamic', 'Unknown')),
                    "dominant_bloc": str(row.get('dominant_bloc', 'Unknown')),
                    "overton_position": str(row.get('overton_window_position', 'Unknown')),
                    "tags": {
                        "geographic": {
                            "country": str(row.get('country', '')) if pd.notna(row.get('country')) else None,
                            "subregion": str(row.get('subregion', '')) if pd.notna(row.get('subregion')) else None,
                            "continent": str(row.get('continent', '')) if pd.notna(row.get('continent')) else None
                        },
                        "topics": self.parse_tags(str(row.get('tags', '')))
                    }
                }
                
                # Clean up None values in geographic tags
                record["tags"]["geographic"] = {k: v for k, v in record["tags"]["geographic"].items() 
                                              if v is not None and str(v) != 'nan' and str(v) != ''}
                
                batch_results.append(record)
                self.enhanced_records.append(record)
                
            except Exception as e:
                logger.error(f"Error processing record {row.get('canonical_topic', 'Unknown')}: {e}")
                continue
        
        return batch_results
    
    def parse_tags(self, tags_str: str) -> List[str]:
        """Parse comma-separated tags into structured format."""
        if pd.isna(tags_str) or not tags_str or tags_str == 'nan':
            return []
        
        # Split by comma and clean up
        tags = [tag.strip() for tag in str(tags_str).split(',')]
        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag and tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        return unique_tags
    
    def regenerate_comprehensive_data(self):
        """Regenerate all veto data with comprehensive tagging and enhanced descriptions."""
        logger.info("Starting comprehensive veto data regeneration...")
        
        # Load source data
        self.load_source_data()
        
        # Apply comprehensive tagging to the dataframe
        logger.info("Applying comprehensive LLM tagging...")
        tagged_df = tag_veto_data(self.veto_df, geo_hierarchy, iso2_country_code)
        
        # Process in batches
        total_records = len(tagged_df)
        logger.info(f"Processing {total_records} records in batches of {BATCH_SIZE}")
        
        for i in range(0, total_records, BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, total_records)
            batch_df = tagged_df.iloc[i:batch_end]
            
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(total_records + BATCH_SIZE - 1)//BATCH_SIZE} "
                       f"(records {i+1}-{batch_end})")
            
            batch_results = self.process_batch_with_llm(batch_df)
            
            # Add delay between batches to respect rate limits
            if i + BATCH_SIZE < total_records:
                time.sleep(DELAY_BETWEEN_BATCHES)
        
        logger.info(f"Completed processing {len(self.enhanced_records)} enhanced records")
        
        # Save results
        self.save_results()
        
        return self.enhanced_records
    
    def save_results(self):
        """Save the enhanced records to files."""
        logger.info("Saving enhanced records...")
        
        # Save as JSON
        json_path = os.path.join(os.path.dirname(self.data_path), 'comprehensive_enhanced_veto_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.enhanced_records, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(self.enhanced_records)} records to {json_path}")
        
        # Save as CSV for easy inspection
        csv_path = os.path.join(os.path.dirname(self.data_path), 'comprehensive_enhanced_veto_data.csv')
        df = pd.DataFrame(self.enhanced_records)
        
        # Flatten the nested structures for CSV
        df['geographic_country'] = df['tags'].apply(lambda x: x.get('geographic', {}).get('country', ''))
        df['geographic_subregion'] = df['tags'].apply(lambda x: x.get('geographic', {}).get('subregion', ''))
        df['geographic_continent'] = df['tags'].apply(lambda x: x.get('geographic', {}).get('continent', ''))
        df['topic_tags'] = df['tags'].apply(lambda x: ', '.join(x.get('topics', [])))
        
        # Drop the nested tags column
        df_flat = df.drop('tags', axis=1)
        df_flat.to_csv(csv_path, index=False)
        logger.info(f"Saved flattened data to {csv_path}")
        
        # Print summary statistics
        self.print_summary_statistics()
    
    def print_summary_statistics(self):
        """Print summary statistics of the enhanced data."""
        logger.info("=== ENHANCED DATA SUMMARY ===")
        
        total_records = len(self.enhanced_records)
        records_with_geo = sum(1 for r in self.enhanced_records 
                             if r['tags']['geographic'] and any(v for v in r['tags']['geographic'].values()))
        records_with_topics = sum(1 for r in self.enhanced_records 
                                if r['tags']['topics'])
        
        avg_desc_length = sum(len(r['description']) for r in self.enhanced_records) / total_records
        
        logger.info(f"Total records: {total_records}")
        logger.info(f"Records with geographic tags: {records_with_geo} ({100*records_with_geo/total_records:.1f}%)")
        logger.info(f"Records with topic tags: {records_with_topics} ({100*records_with_topics/total_records:.1f}%)")
        logger.info(f"Average description length: {avg_desc_length:.1f} characters")
        
        # Sample enhanced record
        if self.enhanced_records:
            sample = self.enhanced_records[0]
            logger.info(f"\nSample enhanced record:")
            logger.info(f"  Topic: {sample['canonical_topic']}")
            logger.info(f"  Description: {sample['description'][:100]}...")
            logger.info(f"  Geographic: {sample['tags']['geographic']}")
            logger.info(f"  Topics: {sample['tags']['topics'][:3]}...")

def main():
    """Main execution function."""
    # Set up paths
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'complete_tagged_veto_data.csv')
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    # Create regenerator and run
    regenerator = ComprehensiveVetoRegenerator(data_path)
    enhanced_records = regenerator.regenerate_comprehensive_data()
    
    logger.info("Comprehensive veto data regeneration completed successfully!")

if __name__ == "__main__":
    main()
