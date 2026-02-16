"""
SEC EDGAR API Client

Fetches historical quarterly financials from SEC EDGAR's free XBRL API.
This enables point-in-time fundamental analysis without look-ahead bias.

SEC EDGAR API Documentation:
https://www.sec.gov/edgar/sec-api-documentation
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# SEC rate limit: 10 requests per second
SEC_REQUEST_DELAY = 0.15  # 150ms between requests


class CIKMapper:
    """Maps ticker symbols to SEC CIK (Central Index Key) numbers."""

    # SEC maintains a JSON file mapping tickers to CIKs
    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(self, cache_db: str, user_agent: str):
        self.cache_db = cache_db
        self.user_agent = user_agent
        self._cache: Dict[str, str] = {}
        self._load_cache()

    def _get_db_connection(self) -> sqlite3.Connection:
        """Get database connection, creating tables if needed."""
        # Ensure directory exists
        Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cik_mapping (
                ticker TEXT PRIMARY KEY,
                cik TEXT NOT NULL,
                company_name TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        return conn

    def _load_cache(self):
        """Load CIK mappings from SQLite cache."""
        try:
            conn = self._get_db_connection()
            cursor = conn.execute("SELECT ticker, cik FROM cik_mapping")
            self._cache = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to load CIK cache: {e}")
            self._cache = {}

    def _save_to_cache(self, ticker: str, cik: str, company_name: str = ""):
        """Save a single CIK mapping to cache."""
        try:
            conn = self._get_db_connection()
            conn.execute(
                """INSERT OR REPLACE INTO cik_mapping (ticker, cik, company_name, updated_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                (ticker.upper(), cik, company_name)
            )
            conn.commit()
            conn.close()
            self._cache[ticker.upper()] = cik
        except Exception as e:
            logger.warning(f"Failed to save CIK mapping: {e}")

    def _fetch_all_mappings(self) -> bool:
        """Fetch all ticker-to-CIK mappings from SEC."""
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(self.COMPANY_TICKERS_URL, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            # SEC format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"}, ...}
            conn = self._get_db_connection()

            for item in data.values():
                ticker = item.get("ticker", "").upper()
                cik_num = item.get("cik_str")
                title = item.get("title", "")

                if ticker and cik_num:
                    # CIK needs to be zero-padded to 10 digits
                    cik = str(cik_num).zfill(10)
                    conn.execute(
                        """INSERT OR REPLACE INTO cik_mapping (ticker, cik, company_name, updated_at)
                           VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                        (ticker, cik, title)
                    )
                    self._cache[ticker] = cik

            conn.commit()
            conn.close()
            logger.info(f"Loaded {len(self._cache)} CIK mappings from SEC")
            return True

        except Exception as e:
            logger.error(f"Failed to fetch CIK mappings: {e}")
            return False

    def get_cik(self, ticker: str) -> Optional[str]:
        """Get CIK for a ticker symbol."""
        ticker = ticker.upper()

        # Check memory cache
        if ticker in self._cache:
            return self._cache[ticker]

        # Try loading from database
        try:
            conn = self._get_db_connection()
            cursor = conn.execute(
                "SELECT cik FROM cik_mapping WHERE ticker = ?", (ticker,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                self._cache[ticker] = row[0]
                return row[0]
        except Exception:
            pass

        # Fetch all mappings if we don't have them
        if len(self._cache) < 1000:  # Likely haven't loaded yet
            if self._fetch_all_mappings():
                return self._cache.get(ticker)

        return None

    def is_sec_filer(self, ticker: str) -> bool:
        """Check if a ticker files with SEC (i.e., not an ETF or foreign stock)."""
        return self.get_cik(ticker) is not None


class SECEdgarClient:
    """
    Client for SEC EDGAR XBRL API.

    Fetches company facts (quarterly financials) with rate limiting and caching.
    """

    # Base URL for company facts
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    # XBRL concepts we need for fundamental analysis
    XBRL_CONCEPTS = {
        # Income Statement
        "revenues": [
            "us-gaap:Revenues",
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:SalesRevenueNet",
            "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        ],
        "net_income": [
            "us-gaap:NetIncomeLoss",
            "us-gaap:ProfitLoss",
            "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
        ],
        "operating_income": [
            "us-gaap:OperatingIncomeLoss",
            "us-gaap:IncomeLossFromContinuingOperationsBeforeInterestExpenseInterestIncomeIncomeTaxesExtraordinaryItemsNoncontrollingInterestsNet",
        ],
        "gross_profit": [
            "us-gaap:GrossProfit",
        ],
        # Balance Sheet
        "total_assets": [
            "us-gaap:Assets",
        ],
        "total_liabilities": [
            "us-gaap:Liabilities",
            "us-gaap:LiabilitiesAndStockholdersEquity",
        ],
        "stockholders_equity": [
            "us-gaap:StockholdersEquity",
            "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        "current_assets": [
            "us-gaap:AssetsCurrent",
        ],
        "current_liabilities": [
            "us-gaap:LiabilitiesCurrent",
        ],
        # Per Share
        "eps_basic": [
            "us-gaap:EarningsPerShareBasic",
        ],
        "eps_diluted": [
            "us-gaap:EarningsPerShareDiluted",
        ],
        "shares_outstanding": [
            "us-gaap:CommonStockSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
        ],
    }

    def __init__(self, cache_db: str, user_agent: str):
        self.cache_db = cache_db
        self.user_agent = user_agent
        self.cik_mapper = CIKMapper(cache_db, user_agent)
        self._last_request_time = 0
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for caching."""
        Path(self.cache_db).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS quarterly_financials (
                ticker TEXT,
                period_end DATE,
                filing_date DATE,
                form TEXT,
                revenues REAL,
                net_income REAL,
                operating_income REAL,
                gross_profit REAL,
                total_assets REAL,
                total_liabilities REAL,
                stockholders_equity REAL,
                current_assets REAL,
                current_liabilities REAL,
                eps_basic REAL,
                eps_diluted REAL,
                shares_outstanding REAL,
                raw_json TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, period_end)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_period
            ON quarterly_financials(ticker, period_end)
        """)
        conn.commit()
        conn.close()

    def _rate_limit(self):
        """Enforce SEC rate limit (10 req/sec)."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < SEC_REQUEST_DELAY:
            time.sleep(SEC_REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _parse_concept_values(
        self, facts: Dict, concept_keys: List[str], is_quarterly: bool = True
    ) -> List[Dict]:
        """
        Parse XBRL concept values from company facts.

        Returns list of {period_end, value, filed, form} sorted by period_end.
        """
        results = []
        seen_periods = set()

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        for concept_key in concept_keys:
            # Remove namespace prefix
            concept_name = concept_key.split(":")[-1]
            concept_data = us_gaap.get(concept_name, {})
            units = concept_data.get("units", {})

            # Try USD first, then shares
            values = units.get("USD", []) or units.get("shares", []) or units.get("USD/shares", [])

            for item in values:
                # Filter for 10-Q (quarterly) or 10-K (annual)
                form = item.get("form", "")
                if is_quarterly and form not in ("10-Q", "10-K"):
                    continue

                period_end = item.get("end")
                if not period_end:
                    continue

                # Skip duplicates
                if period_end in seen_periods:
                    continue

                # For quarterly data, we want items with ~3 month periods
                # Annual reports (10-K) cover the full year
                start = item.get("start")
                if start and form == "10-Q":
                    try:
                        start_dt = datetime.strptime(start, "%Y-%m-%d")
                        end_dt = datetime.strptime(period_end, "%Y-%m-%d")
                        days = (end_dt - start_dt).days
                        # Skip if not approximately quarterly (80-100 days)
                        if not (70 <= days <= 110):
                            continue
                    except ValueError:
                        pass

                seen_periods.add(period_end)
                results.append({
                    "period_end": period_end,
                    "value": item.get("val"),
                    "filed": item.get("filed"),
                    "form": form,
                    "start": start,
                })

        # Sort by period end date
        results.sort(key=lambda x: x["period_end"])
        return results

    def _fetch_company_facts(self, ticker: str) -> Optional[Dict]:
        """Fetch all company facts from SEC EDGAR."""
        cik = self.cik_mapper.get_cik(ticker)
        if not cik:
            logger.warning(f"{ticker}: No CIK found - likely ETF or foreign stock")
            return None

        self._rate_limit()

        url = self.COMPANY_FACTS_URL.format(cik=cik)
        headers = {"User-Agent": self.user_agent}

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"{ticker}: No SEC filings found")
            else:
                logger.error(f"{ticker}: SEC API error: {e}")
            return None
        except Exception as e:
            logger.error(f"{ticker}: Failed to fetch SEC data: {e}")
            return None

    def get_quarterly_financials(
        self, ticker: str, force_refresh: bool = False
    ) -> List[Dict]:
        """
        Get all quarterly financials for a ticker.

        Returns list of quarterly reports sorted by period_end, each containing:
        - period_end: End date of the quarter
        - filing_date: When the report was filed (important for point-in-time)
        - revenues, net_income, operating_income, gross_profit
        - total_assets, total_liabilities, stockholders_equity
        - current_assets, current_liabilities
        - eps_basic, eps_diluted, shares_outstanding
        """
        ticker = ticker.upper()

        # Check cache first
        if not force_refresh:
            cached = self._get_cached_financials(ticker)
            if cached:
                return cached

        # Fetch from SEC
        facts = self._fetch_company_facts(ticker)
        if not facts:
            return []

        # Parse all concepts
        parsed = {}
        for metric, concepts in self.XBRL_CONCEPTS.items():
            parsed[metric] = self._parse_concept_values(facts, concepts)

        # Build quarterly records by period
        quarterly_records = {}

        # Use revenues as the primary driver for periods
        revenue_periods = {r["period_end"]: r for r in parsed.get("revenues", [])}

        # If no revenue data, try net_income
        if not revenue_periods:
            revenue_periods = {r["period_end"]: r for r in parsed.get("net_income", [])}

        for period_end, rev_data in revenue_periods.items():
            record = {
                "ticker": ticker,
                "period_end": period_end,
                "filing_date": rev_data.get("filed"),
                "form": rev_data.get("form"),
            }

            # Add all metrics
            for metric, values in parsed.items():
                # Find value for this period
                metric_val = None
                for v in values:
                    if v["period_end"] == period_end:
                        metric_val = v["value"]
                        break
                record[metric] = metric_val

            quarterly_records[period_end] = record

        # Sort by period and convert to list
        results = [
            quarterly_records[p]
            for p in sorted(quarterly_records.keys())
        ]

        # Cache results
        self._cache_financials(ticker, results)

        return results

    def _get_cached_financials(self, ticker: str) -> Optional[List[Dict]]:
        """Get financials from cache if recent enough."""
        try:
            conn = sqlite3.connect(self.cache_db)
            conn.row_factory = sqlite3.Row

            cursor = conn.execute(
                """SELECT * FROM quarterly_financials
                   WHERE ticker = ?
                   ORDER BY period_end""",
                (ticker.upper(),)
            )
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return None

            # Check if cache is recent (within 7 days)
            latest = rows[-1]
            updated = datetime.fromisoformat(latest["updated_at"])
            if datetime.now() - updated > timedelta(days=7):
                return None

            return [dict(row) for row in rows]

        except Exception as e:
            logger.warning(f"Cache read error for {ticker}: {e}")
            return None

    def _cache_financials(self, ticker: str, records: List[Dict]):
        """Cache financials to SQLite."""
        if not records:
            return

        try:
            conn = sqlite3.connect(self.cache_db)

            for rec in records:
                conn.execute(
                    """INSERT OR REPLACE INTO quarterly_financials
                       (ticker, period_end, filing_date, form,
                        revenues, net_income, operating_income, gross_profit,
                        total_assets, total_liabilities, stockholders_equity,
                        current_assets, current_liabilities,
                        eps_basic, eps_diluted, shares_outstanding,
                        updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (
                        rec.get("ticker"),
                        rec.get("period_end"),
                        rec.get("filing_date"),
                        rec.get("form"),
                        rec.get("revenues"),
                        rec.get("net_income"),
                        rec.get("operating_income"),
                        rec.get("gross_profit"),
                        rec.get("total_assets"),
                        rec.get("total_liabilities"),
                        rec.get("stockholders_equity"),
                        rec.get("current_assets"),
                        rec.get("current_liabilities"),
                        rec.get("eps_basic"),
                        rec.get("eps_diluted"),
                        rec.get("shares_outstanding"),
                    )
                )

            conn.commit()
            conn.close()
            logger.debug(f"Cached {len(records)} quarters for {ticker}")

        except Exception as e:
            logger.warning(f"Cache write error for {ticker}: {e}")

    def get_financials_at_date(
        self, ticker: str, as_of_date: str
    ) -> Optional[Dict]:
        """
        Get the most recent financials available as of a specific date.

        This is the key method for point-in-time analysis - it returns
        only data that was FILED before the as_of_date, avoiding look-ahead bias.

        Args:
            ticker: Stock symbol
            as_of_date: Date string (YYYY-MM-DD) - only use filings before this date

        Returns:
            Dict with most recent quarterly data, or None if unavailable
        """
        quarters = self.get_quarterly_financials(ticker)
        if not quarters:
            return None

        # Filter to only filings available before as_of_date
        available = [
            q for q in quarters
            if q.get("filing_date") and q["filing_date"] <= as_of_date
        ]

        if not available:
            return None

        # Return most recent available
        return available[-1]

    def is_sec_covered(self, ticker: str) -> bool:
        """Check if ticker is covered by SEC (US domestic stocks only)."""
        return self.cik_mapper.is_sec_filer(ticker)


# Singleton instance
_client: Optional[SECEdgarClient] = None


def get_sec_client() -> SECEdgarClient:
    """Get or create the SEC EDGAR client singleton."""
    global _client
    if _client is None:
        from config import SEC_EDGAR_CACHE_DB, SEC_EDGAR_USER_AGENT
        _client = SECEdgarClient(SEC_EDGAR_CACHE_DB, SEC_EDGAR_USER_AGENT)
    return _client


def is_sec_covered(ticker: str) -> bool:
    """Check if ticker files with SEC."""
    return get_sec_client().is_sec_covered(ticker)


def get_quarterly_financials(ticker: str) -> List[Dict]:
    """Get all quarterly financials for a ticker."""
    return get_sec_client().get_quarterly_financials(ticker)


def get_financials_at_date(ticker: str, as_of_date: str) -> Optional[Dict]:
    """Get financials available as of a specific date."""
    return get_sec_client().get_financials_at_date(ticker, as_of_date)
