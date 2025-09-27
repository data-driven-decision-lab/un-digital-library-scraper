"""
Security Council Data Loader

This module handles loading and path resolution for Security Council data files.
"""

import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SecurityCouncilDataLoader:
    """Handles loading of Security Council data files."""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            base_path: Base path to the Security Council data directory
        """
        self.base_path = base_path or self._get_default_data_path()
    
    def _get_default_data_path(self) -> str:
        """Get the default data path based on the current file location."""
        current_file = os.path.abspath(__file__)
        # Go up from services/ to app/ to get the sc_data directory
        services_dir = os.path.dirname(current_file)
        app_dir = os.path.dirname(services_dir)
        return os.path.join(app_dir, 'sc_data')
    
    def get_enhanced_veto_descriptions_path(self) -> str:
        """Get the path to the enhanced veto descriptions JSON file."""
        return os.path.join(self.base_path, 'enhanced_veto_descriptions.json')

    def get_final_analysis_data_path(self) -> str:
        """Get the path to the final analysis data CSV file."""
        return os.path.join(self.base_path, 'final_analysis_data.csv')
    
    def get_research_report_paths(self) -> tuple[str, str]:
        """
        Get the paths to research report files (fixed and original).
        
        Returns:
            Tuple of (fixed_report_path, original_report_path)
        """
        # Use local sc_data directory
        fixed_path = os.path.join(self.base_path, 'researcher_topic_report_fixed.json')
        original_path = os.path.join(self.base_path, 'researcher_topic_report.json')
        return fixed_path, original_path
    
    def get_policy_report_data_path(self) -> str:
        """Get the path to the preprocessed policy report data JSON file."""
        # Use local sc_data directory
        return os.path.join(self.base_path, 'policy_report_data.json')
    
    def get_vote_analysis_data_path(self) -> str:
        """Get the path to the complete UN votes dataset for vote analysis."""
        # Use the UN votes dataset that contains actual voting patterns
        return os.path.join(self.base_path, 'un_votes_with_sc_rows (2).csv')
    
    def file_exists(self, file_path: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(file_path)
    
    def get_available_data_file(self) -> Optional[str]:
        """
        Get the best available data file for analysis.
        
        Returns:
            Path to the best available data file, or None if none found
        """
        logger.info(f"Looking for data files in base path: {self.base_path}")
        
        # Prioritize the detailed JSON file
        enhanced_path = self.get_enhanced_veto_descriptions_path()
        logger.info(f"Checking for enhanced descriptions path: {enhanced_path}")
        if self.file_exists(enhanced_path):
            logger.info("Found enhanced veto descriptions JSON file.")
            return enhanced_path
            
        # Try final analysis data first
        final_analysis_path = self.get_final_analysis_data_path()
        logger.info(f"Checking final analysis path: {final_analysis_path}")
        if self.file_exists(final_analysis_path):
            logger.info("Found final analysis data file")
            return final_analysis_path
        
        # Try fixed research report
        fixed_path, original_path = self.get_research_report_paths()
        logger.info(f"Checking fixed report path: {fixed_path}")
        if self.file_exists(fixed_path):
            logger.info("Found fixed research report file")
            return fixed_path
        
        # Try original research report
        logger.info(f"Checking original report path: {original_path}")
        if self.file_exists(original_path):
            logger.info("Found original research report file")
            return original_path
        
        logger.error("No Security Council data files found")
        return None