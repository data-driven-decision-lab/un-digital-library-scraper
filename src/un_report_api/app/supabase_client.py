"""Supabase client for UN Report API."""

import os
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class SupabaseDataLoader:
    """Handles data loading from Supabase for the UN Report API."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        # For development, allow fallback to hardcoded values
        if not self.supabase_url:
            self.supabase_url = "https://gjakiqtayqltssvbzasd.supabase.co"
            logger.warning("SUPABASE_URL not set, using development default")
        
        if not self.supabase_key:
            self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdqYWtpcXRheXFsdHNzdmJ6YXNkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjkxOTU4OCwiZXhwIjoyMDU4NDk1NTg4fQ.wY8akPd9J-aRVQAOwTiFuOPxWM90fvkvXpyEfPogyfw"
            logger.warning("SUPABASE_KEY not set, using development default")
        
        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")
    
    def load_annual_scores(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load annual scores data from Supabase with pagination support."""
        try:
            all_data = []
            page_size = 1000
            offset = 0
            
            while True:
                query = self.client.table('annual_scores').select('*')
                
                if year is not None:
                    query = query.eq('Year', year)
                
                query = query.range(offset, offset + page_size - 1)
                response = query.execute()
                
                if not response.data:
                    break
                    
                all_data.extend(response.data)
                
                # If we got fewer records than page_size, we've reached the end
                if len(response.data) < page_size:
                    break
                    
                offset += page_size
            
            if all_data:
                df = pd.DataFrame(all_data)
                logger.info(f"Successfully loaded {len(df)} rows from annual_scores table")
                return df
            else:
                logger.warning("No data found in annual_scores table")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading annual_scores from Supabase: {e}")
            raise
    
    def load_pairwise_similarity(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load pairwise similarity data from Supabase."""
        try:
            query = self.client.table('pairwise_similarity_yearly').select('*')
            
            if year is not None:
                query = query.eq('Year', year)
            
            response = query.execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                logger.info(f"Successfully loaded {len(df)} rows from pairwise_similarity_yearly table")
                return df
            else:
                logger.warning("No data found in pairwise_similarity_yearly table")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading pairwise_similarity_yearly from Supabase: {e}")
            raise
    
    def load_topic_votes(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load topic votes data from Supabase."""
        try:
            query = self.client.table('topic_votes_yearly').select('*')
            
            if year is not None:
                query = query.eq('Year', year)
            
            response = query.execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                logger.info(f"Successfully loaded {len(df)} rows from topic_votes_yearly table")
                return df
            else:
                logger.warning("No data found in topic_votes_yearly table")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading topic_votes_yearly from Supabase: {e}")
            raise
    
    def load_country_classifications(self) -> pd.DataFrame:
        """Load country classifications data from Supabase."""
        try:
            # For now, we'll load from the reference data table or create a simple mapping
            # This might need to be adjusted based on your actual Supabase schema
            query = self.client.table('country_classifications_2023').select('*')
            response = query.execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                logger.info(f"Successfully loaded {len(df)} rows from country_classifications_2023 table")
                return df
            else:
                logger.warning("No data found in country_classifications_2023 table")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading country_classifications_2023 from Supabase: {e}")
            # Return empty DataFrame if table doesn't exist
            return pd.DataFrame()
    
    def load_un_region_mapping(self) -> pd.DataFrame:
        """Load UN region mapping data from Supabase."""
        try:
            # This might need to be adjusted based on your actual Supabase schema
            query = self.client.table('un_country_region_mapping').select('*')
            response = query.execute()
            
            if response.data:
                df = pd.DataFrame(response.data)
                logger.info(f"Successfully loaded {len(df)} rows from un_country_region_mapping table")
                return df
            else:
                logger.warning("No data found in un_country_region_mapping table")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading un_country_region_mapping from Supabase: {e}")
            # Return empty DataFrame if table doesn't exist
            return pd.DataFrame()

# Global instance
supabase_loader = SupabaseDataLoader()
