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
LOOKBACK_DAYS_SHORT = 365        # 1 year for short-term predictions (1-7 days)
LOOKBACK_DAYS_LONG = 365 * 3     # 3 years for long-term predictions (8+ days)
MIN_DATA_POINTS = 50             # Reduced to allow more stocks to pass

# Stocks and ETFs to scan (100 total)
STOCK_LIST = [
    # === STOCKS (50) ===
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

    # === ISLAMIC / SHARIAH-COMPLIANT ETFs (US-listed) ===
    "HLAL",   # Wahed FTSE USA Shariah ETF
    "SPUS",   # SP Funds S&P 500 Sharia Industry Exclusions ETF
    "SPRE",   # SP Funds S&P Global REIT Sharia ETF
    "SPSK",   # SP Funds Dow Jones Global Sukuk ETF
    "UMMA",   # Wahed Dow Jones Islamic World ETF
    "SPTE",   # SP Funds S&P Global Technology ETF (Shariah)

    # === ISLAMIC ETFs (London-listed, Yahoo Finance format) ===
    "ISDU.L",  # iShares MSCI USA Islamic UCITS ETF
    "ISDE.L",  # iShares MSCI EM Islamic UCITS ETF
    "ISWD.L",  # iShares MSCI World Islamic UCITS ETF
    "IUSE.L",  # iShares MSCI USA Islamic UCITS ETF (USD)
    "IQSA.L",  # iShares MSCI Saudi Arabia Capped UCITS ETF

    # === HSBC ETFs (London-listed) ===
    "HMWO.L",  # HSBC MSCI World UCITS ETF
    "HMEU.L",  # HSBC MSCI Europe UCITS ETF
    "HMUS.L",  # HSBC MSCI USA UCITS ETF
    "HMEM.L",  # HSBC MSCI EM UCITS ETF
    "HMJP.L",  # HSBC MSCI Japan UCITS ETF

    # === POPULAR ETFs (US-listed) ===
    # Index ETFs
    "SPY",    # S&P 500
    "QQQ",    # Nasdaq 100
    "IWM",    # Russell 2000
    "DIA",    # Dow Jones
    "VTI",    # Total Stock Market
    "VOO",    # Vanguard S&P 500

    # Sector ETFs
    "XLK",    # Technology Select
    "XLF",    # Financial Select
    "XLV",    # Healthcare Select
    "XLE",    # Energy Select
    "XLI",    # Industrial Select
    "XLY",    # Consumer Discretionary

    # International ETFs
    "EFA",    # MSCI EAFE (Developed Markets)
    "EEM",    # MSCI Emerging Markets
    "VWO",    # Vanguard Emerging Markets
    "FXI",    # China Large-Cap
    "EWJ",    # Japan
    "EWG",    # Germany

    # Bond ETFs
    "BND",    # Total Bond Market
    "TLT",    # 20+ Year Treasury
    "HYG",    # High Yield Corporate

    # Commodity ETFs
    "GLD",    # Gold
    "SLV",    # Silver
    "USO",    # Oil

    # Thematic ETFs
    "ARKK",   # ARK Innovation
    "ICLN",   # Clean Energy
    "SOXX",   # Semiconductor

    # === HIGH GROWTH SMALL/MID CAPS (Shariah-compliant) ===
    # Note: These avoid alcohol, tobacco, gambling, pork, weapons, and conventional finance

    # Cloud & Cybersecurity
    "NET",    # Cloudflare
    "CRWD",   # CrowdStrike
    "ZS",     # Zscaler
    "S",      # SentinelOne
    "DDOG",   # Datadog
    "SNOW",   # Snowflake
    "MDB",    # MongoDB
    "ESTC",   # Elastic
    "CFLT",   # Confluent
    "GTLB",   # GitLab

    # AI & Software
    "PLTR",   # Palantir
    "AI",     # C3.ai
    "PATH",   # UiPath
    "HUBS",   # HubSpot
    "TTD",    # Trade Desk
    "U",      # Unity Software
    "RBLX",   # Roblox
    "DUOL",   # Duolingo
    "ASAN",   # Asana
    "DOCN",   # DigitalOcean

    # Electric Vehicles & Clean Energy
    "RIVN",   # Rivian
    "LCID",   # Lucid Motors
    "NIO",    # NIO (China EV)
    "XPEV",   # XPeng
    "LI",     # Li Auto
    "CHPT",   # ChargePoint
    "PLUG",   # Plug Power
    "FSLR",   # First Solar
    "ENPH",   # Enphase Energy
    "SEDG",   # SolarEdge
    "RUN",    # Sunrun
    "BE",     # Bloom Energy
    "STEM",   # Stem Inc

    # Semiconductors (Small/Mid)
    "MRVL",   # Marvell Technology
    "ON",     # ON Semiconductor
    "WOLF",   # Wolfspeed
    "LSCC",   # Lattice Semiconductor
    "CRUS",   # Cirrus Logic
    "SITM",   # SiTime

    # Healthcare Tech (Shariah-compliant)
    "DXCM",   # DexCom (glucose monitors)
    "ISRG",   # Intuitive Surgical (robotics)
    "VEEV",   # Veeva Systems
    "DOCS",   # Doximity
    "HIMS",   # Hims & Hers Health

    # E-commerce & Consumer Tech
    "SHOP",   # Shopify
    "ETSY",   # Etsy
    "CHWY",   # Chewy
    "SE",     # Sea Limited
    "MELI",   # MercadoLibre

    # Space & Communications
    "RKLB",   # Rocket Lab
    "ASTS",   # AST SpaceMobile
    "IRDM",   # Iridium
    "IONQ",   # IonQ (Quantum Computing)
]
