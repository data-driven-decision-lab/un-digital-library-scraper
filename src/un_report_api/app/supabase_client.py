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
        """Load annual scores data from local CSV file."""
        try:
            # Load from local CSV file in required_csvs folder
            csv_path = os.path.join(os.path.dirname(__file__), 'required_csvs', 'annual_scores.csv')
            
            if not os.path.exists(csv_path):
                logger.error(f"Annual scores CSV file not found at: {csv_path}")
                raise FileNotFoundError(f"Annual scores CSV file not found at: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            if year is not None:
                df = df[df['Year'] == year]
            
            if not df.empty:
                logger.info(f"Successfully loaded {len(df)} rows from annual_scores.csv")
                return df
            else:
                logger.warning("No data found in annual_scores.csv")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading annual_scores from CSV: {e}")
            raise
    
    def load_pairwise_similarity(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load pairwise similarity data from local CSV file."""
        try:
            # Load from local CSV file in required_csvs folder
            csv_path = os.path.join(os.path.dirname(__file__), 'required_csvs', 'pairwise_similarity_yearly.csv')
            
            if not os.path.exists(csv_path):
                logger.error(f"Pairwise similarity CSV file not found at: {csv_path}")
                raise FileNotFoundError(f"Pairwise similarity CSV file not found at: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            if year is not None:
                df = df[df['Year'] == year]
            
            if not df.empty:
                logger.info(f"Successfully loaded {len(df)} rows from pairwise_similarity_yearly.csv")
                return df
            else:
                logger.warning("No data found in pairwise_similarity_yearly.csv")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading pairwise_similarity_yearly from CSV: {e}")
            raise
    
    def load_topic_votes(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load topic votes data from local CSV file."""
        try:
            # Load from local CSV file in required_csvs folder
            csv_path = os.path.join(os.path.dirname(__file__), 'required_csvs', 'topic_votes_yearly.csv')
            
            if not os.path.exists(csv_path):
                logger.error(f"Topic votes CSV file not found at: {csv_path}")
                raise FileNotFoundError(f"Topic votes CSV file not found at: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            if year is not None:
                df = df[df['Year'] == year]
            
            if not df.empty:
                logger.info(f"Successfully loaded {len(df)} rows from topic_votes_yearly.csv")
                return df
            else:
                logger.warning("No data found in topic_votes_yearly.csv")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading topic_votes_yearly from CSV: {e}")
            raise
    
    def load_country_classifications(self) -> pd.DataFrame:
        """Load country classifications data from local CSV file."""
        try:
            # Load from local CSV file in required_csvs folder
            csv_path = os.path.join(os.path.dirname(__file__), 'required_csvs', 'country_classifications_2023.csv')
            
            if not os.path.exists(csv_path):
                logger.error(f"Country classifications CSV file not found at: {csv_path}")
                raise FileNotFoundError(f"Country classifications CSV file not found at: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            if not df.empty:
                logger.info(f"Successfully loaded {len(df)} rows from country_classifications_2023.csv")
                return df
            else:
                logger.warning("No data found in country_classifications_2023.csv")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading country_classifications_2023 from CSV: {e}")
            # Return empty DataFrame if file doesn't exist
            return pd.DataFrame()
    
    def load_un_region_mapping(self) -> pd.DataFrame:
        """Load UN region mapping data from local CSV file."""
        try:
            # Load from local CSV file in required_csvs folder
            csv_path = os.path.join(os.path.dirname(__file__), 'required_csvs', 'UN_Country_Region_Mapping_clean.csv')
            
            if not os.path.exists(csv_path):
                logger.error(f"UN region mapping CSV file not found at: {csv_path}")
                raise FileNotFoundError(f"UN region mapping CSV file not found at: {csv_path}")
            
            df = pd.read_csv(csv_path)
            
            if not df.empty:
                logger.info(f"Successfully loaded {len(df)} rows from UN_Country_Region_Mapping_clean.csv")
                return df
            else:
                logger.warning("No data found in UN_Country_Region_Mapping_clean.csv")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading UN region mapping from CSV: {e}")
            # Return empty DataFrame if file doesn't exist
            return pd.DataFrame()

# Global instance
supabase_loader = SupabaseDataLoader()
