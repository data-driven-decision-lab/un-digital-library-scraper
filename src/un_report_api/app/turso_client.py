"""Turso (LibSQL) client for UN Report API."""

import json
import os
import logging
import math
import urllib.request
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)


class _TursoHTTPCursor:
    """DB-API-style cursor backed by a single Turso HTTP response."""

    def __init__(self, columns, rows):
        self.description = [(c, None, None, None, None, None, None) for c in columns]
        self._rows = rows
        self._pos = 0

    def fetchall(self):
        return self._rows

    def fetchone(self):
        if self._pos < len(self._rows):
            row = self._rows[self._pos]
            self._pos += 1
            return row
        return None

    @property
    def rowcount(self):
        return len(self._rows)


class _TursoHTTPConnection:
    """SQLite-like connection backed by the Turso HTTP API (v2/pipeline).

    Used as a fallback when libsql_experimental isn't installable
    (e.g. on Windows where the wheel doesn't build). Public surface
    (execute/commit/close) matches libsql.Connection enough for the
    read paths the API uses.
    """

    def __init__(self, url, auth_token):
        self._url = url.replace("libsql://", "https://")
        self._token = auth_token
        self._api_url = f"{self._url}/v2/pipeline"

    @staticmethod
    def _convert_arg(a):
        if a is None:
            return {"type": "null"}
        if hasattr(a, "item"):
            a = a.item()
        if isinstance(a, float) and (math.isnan(a) or math.isinf(a)):
            return {"type": "null"}
        if isinstance(a, bool):
            return {"type": "integer", "value": str(int(a))}
        if isinstance(a, int):
            return {"type": "integer", "value": str(a)}
        if isinstance(a, float):
            return {"type": "float", "value": a}
        return {"type": "text", "value": str(a)}

    def execute(self, sql, args=None):
        stmt = {"sql": sql}
        if args:
            stmt["args"] = [self._convert_arg(a) for a in args]
        payload = json.dumps({"requests": [{"type": "execute", "stmt": stmt}, {"type": "close"}]})
        req = urllib.request.Request(
            self._api_url,
            data=payload.encode(),
            headers={"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=60)
        data = json.loads(resp.read())
        result = data["results"][0]
        if result["type"] == "error":
            raise Exception(f"Turso SQL error: {result.get('error', {}).get('message', 'unknown')}")
        exec_result = result["response"]["result"]
        columns = [col["name"] for col in exec_result.get("cols", [])]
        raw_rows = exec_result.get("rows", [])
        rows = []
        for raw_row in raw_rows:
            row = []
            for cell in raw_row:
                if cell["type"] == "null":
                    row.append(None)
                elif cell["type"] == "integer":
                    row.append(int(cell["value"]))
                elif cell["type"] == "float":
                    row.append(float(cell["value"]))
                else:
                    row.append(cell["value"])
            rows.append(tuple(row))
        return _TursoHTTPCursor(columns, rows)

    def commit(self):
        pass  # Turso auto-commits

    def close(self):
        pass


def get_turso_connection():
    """
    Create a connection to Turso. Prefers libsql_experimental (native client);
    falls back to the HTTP-based client when libsql_experimental is unavailable
    (e.g. on Windows, where the wheel doesn't build). Both clients expose the
    same execute()/commit()/close() interface.

    Imports are deferred inside the function so that ImportError is localised
    to connection attempts rather than raised at module-import time.
    """
    url = os.getenv("TURSO_DATABASE_URL")
    auth_token = os.getenv("TURSO_AUTH_TOKEN")
    if not url:
        raise ValueError("TURSO_DATABASE_URL environment variable not set.")
    if not auth_token:
        raise ValueError("TURSO_AUTH_TOKEN environment variable not set.")
    try:
        import libsql_experimental as libsql  # noqa: PLC0415
        return libsql.connect(url, auth_token=auth_token)
    except ImportError:
        return _TursoHTTPConnection(url, auth_token)


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
