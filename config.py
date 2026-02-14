import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

# Cache TTL in seconds (6 hours - predictions use daily close prices only)
STOCK_DATA_CACHE_TTL = 21600  # 6 hours
PREDICTION_CACHE_TTL = 21600  # 6 hours
SCREENER_CACHE_TTL = 21600    # 6 hours

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
    "WMT", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "HD", "LOW", "BKNG",
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

# Expanded stock universe for long-term composite scoring
# Combines S&P 500 + NASDAQ-100 + current STOCK_LIST (deduplicated)
# ~550 stocks for comprehensive fundamental analysis
COMPOSITE_STOCK_LIST = [
    # === S&P 500 CORE (Top 150 by market cap) ===
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO",
    "ACN", "ABT", "BAC", "CRM", "DHR", "PFE", "CMCSA", "VZ", "ADBE", "NKE",
    "TXN", "NFLX", "WFC", "PM", "ORCL", "AMD", "INTC", "UPS", "RTX", "NEE",
    "HON", "COP", "BMY", "QCOM", "LOW", "UNP", "MS", "SPGI", "INTU", "IBM",
    "ELV", "BA", "AMGN", "GE", "CAT", "DE", "SBUX", "GS", "ISRG", "PLD",
    "GILD", "MDT", "AXP", "BLK", "LMT", "SYK", "BKNG", "ADI", "MDLZ", "CVS",
    "TMUS", "CB", "REGN", "AMT", "SCHW", "TJX", "LRCX", "ADP", "MMC", "MO",
    "CI", "EOG", "SLB", "ZTS", "VRTX", "SO", "DUK", "CME", "SNPS", "CDNS",
    "BDX", "CL", "ITW", "BSX", "EQIX", "FI", "WM", "ETN", "APD", "AON",
    "NOC", "SHW", "ICE", "KLAC", "PNC", "HCA", "CSX", "EMR", "PGR", "MPC",
    "ORLY", "GD", "MCO", "FCX", "MCK", "USB", "AZO", "MAR", "CTAS", "MCHP",
    "NSC", "ROP", "AJG", "MSI", "SRE", "AFL", "PSX", "TT", "APH", "AEP",
    "CMG", "PCAR", "F", "GM", "TRV", "KMB", "WELL", "ADSK", "PAYX", "PSA",

    # === NASDAQ-100 ADDITIONS ===
    "DXCM", "PANW", "MNST", "CPRT", "PYPL", "MELI", "FTNT", "IDXX", "ODFL",
    "FAST", "CSGP", "AEE", "KHC", "VRSK", "DLTR", "CDW", "GIS", "BKR", "EW",
    "CTSH", "BIIB", "KDP", "ON", "TEAM", "ALGN", "WBD", "ENPH", "ANSS", "ZS",
    "DDOG", "CRWD", "CEG", "TTD", "SIRI", "LCID", "RIVN", "FANG", "GEHC",

    # === HIGH GROWTH MID-CAPS (from original list) ===
    # Cloud & Cybersecurity
    "NET", "S", "SNOW", "MDB", "ESTC", "CFLT", "GTLB",
    # AI & Software
    "PLTR", "AI", "PATH", "HUBS", "U", "RBLX", "DUOL", "ASAN", "DOCN",
    # Electric Vehicles & Clean Energy
    "NIO", "XPEV", "LI", "CHPT", "PLUG", "FSLR", "SEDG", "RUN", "BE", "STEM",
    # Semiconductors
    "MRVL", "WOLF", "LSCC", "CRUS", "SITM",
    # Healthcare Tech
    "VEEV", "DOCS", "HIMS",
    # E-commerce & Consumer Tech
    "SHOP", "ETSY", "CHWY", "SE",
    # Space & Communications
    "RKLB", "ASTS", "IRDM", "IONQ",
    # Fintech
    "SQ", "COIN", "AFRM", "UPST", "SOFI", "HOOD",

    # === ADDITIONAL S&P 500 COMPONENTS ===
    "MSCI", "MPWR", "KEYS", "TDG", "FICO", "TRGP", "VLTO", "DECK", "AXON",
    "EXC", "D", "PEG", "ED", "XEL", "ES", "AWK", "WEC", "DTE", "AES",
    "HES", "DVN", "OXY", "HAL", "MRO", "VLO", "PSX", "HUM", "CNC", "MOH",
    "A", "ILMN", "IQV", "TECH", "WAT", "MTD", "PKI", "HOLX", "DGX", "LH",
    "STZ", "DEO", "TAP", "SAM", "BF.B", "MNST", "CCEP", "KDP", "STZ",
    "AIG", "ALL", "MET", "PRU", "TRV", "HIG", "CINF", "GL", "AFG", "WRB",
    "FIS", "GPN", "JKHY", "ADP", "PAYX", "PAYC", "PCTY", "WEX", "BR",
    "CARR", "JCI", "TT", "LII", "GNRC", "HUBB", "EME", "PWR", "FSLR",
    "POOL", "WSO", "SNA", "SWK", "FAST", "GWW", "W.W. Grainger", "NDSN",
    "DOV", "IR", "PH", "ROK", "AME", "ZBRA", "GRMN", "TER", "ENTG",
    "J", "TTEK", "FTV", "BAH", "LDOS", "SAIC", "KBR", "CACI", "BWXT",
    "LKQ", "APTV", "BWA", "ALV", "MGA", "LEA", "GNTX", "VC", "DAN",
    "ROL", "CHRW", "XPO", "EXPD", "JBHT", "SNDR", "LSTR", "SAIA", "WERN",
    "RCL", "CCL", "NCLH", "MAR", "HLT", "H", "WH", "CHH", "IHG",
    "EXPE", "ABNB", "TRIP", "BKNG", "UBER", "LYFT", "DASH", "GRAB",
    "DRI", "CMG", "YUM", "QSR", "WING", "CAVA", "SHAK", "DPZ", "MCD",
    "WEN", "JACK", "PZZA", "EAT", "TXRH", "CAKE", "DIN", "BLMN",

    # === MATERIALS & INDUSTRIALS ===
    "LIN", "APD", "ECL", "SHW", "PPG", "NUE", "STLD", "RS", "CMC", "X",
    "CLF", "AA", "CENX", "KALU", "HUN", "WLK", "OLN", "CE", "EMN", "IFF",
    "ALB", "LIVENT", "LAC", "PLL", "MP", "SGML", "UUUU",

    # === REITS ===
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB",
    "EQR", "VTR", "ARE", "BXP", "SLG", "VNO", "KIM", "REG", "FRT", "NNN",
    "WPC", "STOR", "STAG", "REXR", "FR", "EGP", "COLD", "IIPR",

    # === FINANCIALS ===
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
    "BK", "STT", "NTRS", "AXP", "DFS", "SYF", "ALLY", "CACC", "OMF",
    "CME", "ICE", "NDAQ", "CBOE", "MKTX", "VIRT", "SCHW", "IBKR", "RJF",
    "SF", "EVR", "LAZ", "HLI", "PJT", "MC", "GHL", "JEF", "PIPR",
    "BX", "KKR", "APO", "CG", "ARES", "OWL", "VCTR", "AB", "IVZ",
    "BEN", "TROW", "SEIC", "VRTS", "JHG", "APAM", "FHI", "EV",

    # === HEALTHCARE ===
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "REGN", "VRTX", "BIIB", "MRNA", "BNTX", "SGEN", "BGNE",
    "ALNY", "INCY", "EXEL", "SRPT", "BMRN", "UTHR", "IONS", "NBIX", "PTCT",
    "BSX", "MDT", "SYK", "EW", "ZBH", "ISRG", "HOLX", "DXCM", "PODD", "TFX",
    "STE", "BAX", "BDX", "WAT", "A", "MTD", "PKI", "TECH", "IDXX", "ALGN",
    "HCA", "UHS", "THC", "CYH", "SEM", "ACHC", "ENSG", "NHC", "PNTG",
    "CVS", "WBA", "MCK", "ABC", "CAH", "OMI", "PDCO", "HSIC",

    # === COMMUNICATION SERVICES ===
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "CHTR", "TMUS",
    "EA", "ATVI", "TTWO", "RBLX", "U", "SE", "BILI", "HUYA", "DOYU",
    "MTCH", "BMBL", "PINS", "SNAP", "TWTR", "ZM", "SPOT", "TME", "IQ",
    "WBD", "PARA", "FOXA", "LYV", "MSGS", "WWE", "EDR", "SKLZ",
    "SIRI", "IHRT", "CMN", "NXST", "TGNA", "SSP", "GTN", "NWS", "NYT",

    # === CONSUMER DISCRETIONARY ===
    "AMZN", "TSLA", "HD", "LOW", "NKE", "SBUX", "TJX", "BKNG", "MCD",
    "ROST", "DG", "DLTR", "TGT", "BBY", "ULTA", "LULU", "GPS", "ANF",
    "URBN", "AEO", "EXPR", "CTRN", "PLCE", "RVLV", "FTCH", "REAL",
    "ETSY", "W", "OSTK", "CVNA", "CARG", "VROOM", "SFT", "LOTZ",
    "RH", "WSM", "ARHS", "SNBR", "LOVE", "TPX", "LEG", "ETH",
    "F", "GM", "RIVN", "LCID", "FSR", "NIO", "XPEV", "LI", "GOEV",
    "AN", "PAG", "LAD", "SAH", "ABG", "GPI", "DRVN", "SIX",
    "GRMN", "BC", "PII", "HOG", "THO", "WGO", "LCII", "CWH",
    "HAS", "MAT", "FNKO", "JAKK", "PLBY", "GOLF", "ELY", "MODG",
    "MGM", "WYNN", "LVS", "CZR", "PENN", "DKNG", "RSI", "GNOG",

    # === CONSUMER STAPLES ===
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL", "KMB",
    "EL", "STZ", "GIS", "K", "KHC", "HSY", "HRL", "CPB", "SJM", "MKC",
    "TSN", "CAG", "INGR", "BGS", "POST", "HAIN", "SMPL", "FRPT",
    "CLX", "CHD", "SPB", "SWM", "NWL", "REV", "IPAR", "EPC", "ENR",
    "KR", "ACI", "SFM", "GO", "IMKTA", "CASY", "UNFI", "SPTN", "KDP",
    "MNST", "FIZZ", "CELH", "ZVIA", "BFI", "SAM", "TAP", "STZ",

    # === ENERGY ===
    "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "VLO", "PSX",
    "OXY", "HES", "DVN", "FANG", "HAL", "BKR", "MRO", "APA", "MTDR",
    "OVV", "RRC", "EQT", "AR", "SWN", "CNX", "COG", "CTRA",
    "WMB", "KMI", "OKE", "TRGP", "LNG", "ET", "EPD", "MPLX", "PAA",
    "DK", "PARR", "PBF", "HFC", "CVI", "DINO", "CLMT", "CAPL",

    # === UTILITIES ===
    "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED",
    "WEC", "DTE", "ES", "AES", "ETR", "FE", "PPL", "CMS", "CNP", "AEE",
    "EVRG", "NI", "PNW", "OGE", "LNT", "MGEE", "NWE", "AVA", "BKH",
    "AWK", "WTRG", "CWT", "SJW", "AWR", "YORW", "ARTNA", "MSEX",
]

# Deduplicate the list
COMPOSITE_STOCK_LIST = list(dict.fromkeys(COMPOSITE_STOCK_LIST))
