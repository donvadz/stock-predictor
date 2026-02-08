import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Cache TTL in seconds
STOCK_DATA_CACHE_TTL = 600  # 10 minutes
PREDICTION_CACHE_TTL = 180  # 3 minutes
SCREENER_CACHE_TTL = 300  # 5 minutes

# Model Configuration
LOOKBACK_DAYS = 365
MIN_DATA_POINTS = 100

# Popular stocks to scan (top 50 by market cap + popular trades)
STOCK_LIST = [
    # Tech giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM",
    # Finance
    "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "BLK", "C",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "HD", "LOW",
    # Other
    "XOM", "CVX", "DIS", "NFLX", "PYPL", "UBER", "ABNB", "SQ", "COIN", "SNAP",
]
