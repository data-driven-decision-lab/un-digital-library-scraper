"""Turso (LibSQL) client for UN Report API."""

import os
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


def get_turso_connection():
    """
    Create a libsql_experimental connection to Turso.
    Returns a connection object, or raises if credentials are missing.

    The libsql_experimental import is intentionally deferred inside this
    function so that an ImportError is localised to connection attempts
    rather than raised at module-import time. This allows the API to start
    without libsql_experimental installed (e.g. in CI environments that only
    need the CSV-reading code paths).
    """
    import libsql_experimental as libsql  # noqa: PLC0415
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    if not url:
        raise ValueError("TURSO_DATABASE_URL environment variable not set.")
    if not auth_token:
        raise ValueError("TURSO_AUTH_TOKEN environment variable not set.")
    conn = libsql.connect(url, auth_token=auth_token)
    return conn


class TursoDataLoader:
    """
    Handles data loading for the UN Report API.

    Currently reads from local CSV files (the pipeline writes CSVs after each
    run). The get_turso_connection() helper is available for direct DB queries
    when needed.

    Replaces SupabaseDataLoader — identical public interface. No Supabase SDK
    import anywhere in this module.
    """

    def load_annual_scores(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load annual scores data from local CSV file."""
        try:
            csv_path = os.path.join(
                os.path.dirname(__file__), "required_csvs", "annual_scores.csv"
            )

            if not os.path.exists(csv_path):
                logger.error("Annual scores CSV file not found at: %s", csv_path)
                raise FileNotFoundError(
                    f"Annual scores CSV file not found at: {csv_path}"
                )

            df = pd.read_csv(csv_path)

            if year is not None:
                df = df[df["Year"] == year]

            if not df.empty:
                logger.info(
                    "Successfully loaded %d rows from annual_scores.csv", len(df)
                )
                return df
            else:
                logger.warning("No data found in annual_scores.csv")
                return pd.DataFrame()

        except Exception as e:
            logger.error("Error loading annual_scores from CSV: %s", e)
            raise

    def load_pairwise_similarity(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load pairwise similarity data from local CSV file."""
        try:
            csv_path = os.path.join(
                os.path.dirname(__file__),
                "required_csvs",
                "pairwise_similarity_yearly.csv",
            )

            if not os.path.exists(csv_path):
                logger.error(
                    "Pairwise similarity CSV file not found at: %s", csv_path
                )
                raise FileNotFoundError(
                    f"Pairwise similarity CSV file not found at: {csv_path}"
                )

            df = pd.read_csv(csv_path)

            if year is not None:
                df = df[df["Year"] == year]

            if not df.empty:
                logger.info(
                    "Successfully loaded %d rows from pairwise_similarity_yearly.csv",
                    len(df),
                )
                return df
            else:
                logger.warning("No data found in pairwise_similarity_yearly.csv")
                return pd.DataFrame()

        except Exception as e:
            logger.error("Error loading pairwise_similarity_yearly from CSV: %s", e)
            raise

    def load_topic_votes(self, year: Optional[int] = None) -> pd.DataFrame:
        """Load topic votes data from local CSV file."""
        try:
            csv_path = os.path.join(
                os.path.dirname(__file__),
                "required_csvs",
                "topic_votes_yearly.csv",
            )

            if not os.path.exists(csv_path):
                logger.error("Topic votes CSV file not found at: %s", csv_path)
                raise FileNotFoundError(
                    f"Topic votes CSV file not found at: {csv_path}"
                )

            df = pd.read_csv(csv_path)

            if year is not None:
                df = df[df["Year"] == year]

            if not df.empty:
                logger.info(
                    "Successfully loaded %d rows from topic_votes_yearly.csv", len(df)
                )
                return df
            else:
                logger.warning("No data found in topic_votes_yearly.csv")
                return pd.DataFrame()

        except Exception as e:
            logger.error("Error loading topic_votes_yearly from CSV: %s", e)
            raise

    def load_country_classifications(self) -> pd.DataFrame:
        """Load country classifications data from local CSV file."""
        try:
            csv_path = os.path.join(
                os.path.dirname(__file__),
                "required_csvs",
                "country_classifications_2023.csv",
            )

            if not os.path.exists(csv_path):
                logger.error(
                    "Country classifications CSV file not found at: %s", csv_path
                )
                raise FileNotFoundError(
                    f"Country classifications CSV file not found at: {csv_path}"
                )

            df = pd.read_csv(csv_path)

            if not df.empty:
                logger.info(
                    "Successfully loaded %d rows from country_classifications_2023.csv",
                    len(df),
                )
                return df
            else:
                logger.warning("No data found in country_classifications_2023.csv")
                return pd.DataFrame()

        except Exception as e:
            logger.error("Error loading country_classifications_2023 from CSV: %s", e)
            # Return empty DataFrame if file doesn't exist
            return pd.DataFrame()

    def load_un_region_mapping(self) -> pd.DataFrame:
        """Load UN region mapping data from local CSV file."""
        try:
            csv_path = os.path.join(
                os.path.dirname(__file__),
                "required_csvs",
                "UN_Country_Region_Mapping_clean.csv",
            )

            if not os.path.exists(csv_path):
                logger.error(
                    "UN region mapping CSV file not found at: %s", csv_path
                )
                raise FileNotFoundError(
                    f"UN region mapping CSV file not found at: {csv_path}"
                )

            df = pd.read_csv(csv_path)

            if not df.empty:
                logger.info(
                    "Successfully loaded %d rows from UN_Country_Region_Mapping_clean.csv",
                    len(df),
                )
                return df
            else:
                logger.warning("No data found in UN_Country_Region_Mapping_clean.csv")
                return pd.DataFrame()

        except Exception as e:
            logger.error("Error loading UN region mapping from CSV: %s", e)
            # Return empty DataFrame if file doesn't exist
            return pd.DataFrame()


# Global instance — drop-in replacement for supabase_loader
turso_loader = TursoDataLoader()
