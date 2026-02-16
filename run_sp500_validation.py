#!/usr/bin/env python3
"""
Run grade validation test on S&P 500 stocks across all horizons.

Usage:
    python run_sp500_validation.py                   # Use SEC EDGAR historical data
    python run_sp500_validation.py --yfinance        # Use yfinance (look-ahead bias)
    python run_sp500_validation.py --historical      # Explicitly use SEC EDGAR
"""

import argparse
import time
from datetime import datetime

# S&P 500 stocks (complete list)
SP500_STOCKS = [
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
    "DXCM", "PANW", "MNST", "CPRT", "PYPL", "MELI", "FTNT", "IDXX", "ODFL",
    "FAST", "CSGP", "AEE", "KHC", "VRSK", "DLTR", "CDW", "GIS", "BKR", "EW",
    "CTSH", "BIIB", "KDP", "ON", "TEAM", "ALGN", "WBD", "ENPH", "ANSS", "ZS",
    "DDOG", "CRWD", "CEG", "TTD", "FANG", "GEHC",
    "MSCI", "MPWR", "KEYS", "TDG", "FICO", "TRGP", "VLTO", "DECK", "AXON",
    "EXC", "D", "PEG", "ED", "XEL", "ES", "AWK", "WEC", "DTE", "AES",
    "HES", "DVN", "OXY", "HAL", "MRO", "VLO",
    "A", "ILMN", "IQV", "TECH", "WAT", "MTD", "PKI", "HOLX", "DGX", "LH",
    "STZ", "TAP", "AIG", "ALL", "MET", "PRU", "HIG", "CINF", "GL", "AFG", "WRB",
    "FIS", "GPN", "JKHY", "PAYC", "PCTY", "WEX", "BR",
    "CARR", "JCI", "LII", "GNRC", "HUBB", "EME", "PWR", "FSLR",
    "POOL", "WSO", "SNA", "SWK", "GWW", "NDSN",
    "DOV", "IR", "PH", "ROK", "AME", "ZBRA", "GRMN", "TER", "ENTG",
    "J", "TTEK", "FTV", "BAH", "LDOS", "KBR", "CACI", "BWXT",
    "LKQ", "APTV", "BWA", "MGA", "LEA", "GNTX", "VC",
    "ROL", "CHRW", "XPO", "EXPD", "JBHT", "SAIA",
    "RCL", "CCL", "NCLH", "HLT", "H", "WH",
    "EXPE", "ABNB", "UBER", "LYFT", "DASH",
    "DRI", "YUM", "QSR", "DPZ",
    "LIN", "ECL", "PPG", "NUE", "STLD", "RS", "CLF", "AA",
    "ALB", "MP",
    "CCI", "SPG", "O", "DLR", "AVB",
    "EQR", "VTR", "ARE", "BXP", "KIM", "REG", "FRT", "NNN",
    "WPC", "STAG", "REXR", "FR",
    "C", "TFC", "COF",
    "BK", "STT", "NTRS", "DFS", "SYF", "ALLY",
    "NDAQ", "CBOE", "MKTX", "VIRT", "IBKR", "RJF",
    "SF", "EVR", "LAZ", "HLI", "PJT", "JEF",
    "BX", "KKR", "APO", "CG", "ARES",
    "BEN", "TROW", "IVZ",
    "MRNA", "INCY", "EXEL",
    "PODD", "TFX",
    "STE", "BAX",
    "UHS", "THC", "ACHC", "ENSG",
    "WBA", "ABC", "CAH", "HSIC",
    "DIS", "T", "CHTR",
    "EA", "TTWO",
    "MTCH", "PINS", "SNAP", "ZM", "SPOT",
    "PARA", "LYV",
    "NXST", "NWS", "NYT",
    "ROST", "DG", "TGT", "BBY", "ULTA", "LULU", "GPS", "ANF",
    "W", "CVNA",
    "RH", "WSM",
    "AN", "PAG", "LAD", "GPI",
    "BC", "PII", "HOG", "THO", "WGO",
    "HAS", "MAT", "ELY",
    "MGM", "WYNN", "LVS", "CZR", "PENN", "DKNG",
    "EL", "HSY", "HRL", "CPB", "SJM", "MKC",
    "TSN", "CAG", "POST",
    "CLX", "CHD", "NWL",
    "KR", "ACI", "SFM", "CASY",
    "FIZZ", "CELH",
    "PXD", "MTDR",
    "OVV", "EQT", "AR", "SWN", "CTRA",
    "WMB", "KMI", "OKE", "LNG", "ET", "EPD", "MPLX", "PAA",
    "DK", "PBF",
    "ETR", "FE", "PPL", "CMS", "CNP",
    "EVRG", "NI", "PNW", "LNT",
    "WTRG",
]

# Deduplicate
SP500_STOCKS = list(dict.fromkeys(SP500_STOCKS))

# All horizons to test
HORIZONS = [3, 6, 12, 24, 60, 120]

def run_validation(use_historical_fundamentals: bool = True):
    from composite_backtest_grades import run_walk_forward_grade_validation

    fundamentals_source = "SEC EDGAR (historical)" if use_historical_fundamentals else "yfinance (current)"

    print("=" * 80)
    print("S&P 500 GRADE VALIDATION TEST")
    print(f"Stocks: {len(SP500_STOCKS)} | Horizons: {HORIZONS}")
    print(f"Fundamentals: {fundamentals_source}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = {}

    for horizon in HORIZONS:
        print(f"\n{'='*60}")
        print(f"TESTING {horizon}-MONTH HORIZON")
        print(f"{'='*60}")

        start_time = time.time()

        def progress(pct, total, msg):
            bar_len = 30
            filled = int(bar_len * pct / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{bar}] {pct}% - {msg}", end="", flush=True)

        result = run_walk_forward_grade_validation(
            stocks=SP500_STOCKS,
            horizon_months=horizon,
            test_periods=[10, 5, 2, 1],  # Test 10Y, 5Y, 2Y, 1Y ago
            max_workers=8,
            progress_callback=progress,
            use_historical_fundamentals=use_historical_fundamentals,
        )

        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.1f}s")

        results[horizon] = result

        # Print summary for this horizon
        print(f"\n  VERDICT: {result.get('verdict', 'N/A')}")
        print(f"  {result.get('verdict_detail', '')}")
        print(f"\n  Summary:")
        summary = result.get('summary', {})
        print(f"    A-grade success rate: {summary.get('avg_a_grade_success_rate', 0):.1f}%")
        print(f"    B-grade success rate: {summary.get('avg_b_grade_success_rate', 0):.1f}%")
        print(f"    A+B combined success: {summary.get('avg_ab_combined_success_rate', 0):.1f}%")

        # Factor analysis
        factors = result.get('factor_analysis', {})
        if factors:
            print(f"\n  Factor Analysis:")
            for factor, data in factors.items():
                corr = data.get('correlation', 0)
                power = data.get('predictive_power', 'N/A')
                spread = data.get('spread', 0)
                print(f"    {factor:15s}: r={corr:+.3f} ({power}), spread={spread:+.1f}%")

    # Final summary across all horizons
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL HORIZONS")
    print("=" * 80)

    print(f"\n{'Horizon':<10} {'Verdict':<20} {'A%':<8} {'B%':<8} {'A+B%':<8} {'Stocks':<8} {'SEC':<8}")
    print("-" * 78)

    for horizon in HORIZONS:
        r = results.get(horizon, {})
        verdict = r.get('verdict', 'N/A')
        summary = r.get('summary', {})
        a_rate = summary.get('avg_a_grade_success_rate', 0)
        b_rate = summary.get('avg_b_grade_success_rate', 0)
        ab_rate = summary.get('avg_ab_combined_success_rate', 0)
        stocks = r.get('stocks_tested', 0)
        sec_covered = r.get('sec_covered_stocks', 0)

        print(f"{horizon}M{'':<7} {verdict:<20} {a_rate:<8.1f} {b_rate:<8.1f} {ab_rate:<8.1f} {stocks:<8} {sec_covered:<8}")

    # Best/worst factors by horizon
    print("\n" + "-" * 70)
    print("BEST PREDICTIVE FACTOR BY HORIZON:")
    print("-" * 70)

    for horizon in HORIZONS:
        r = results.get(horizon, {})
        factors = r.get('factor_analysis', {})
        if factors:
            best = max(factors.items(), key=lambda x: x[1].get('correlation', -999))
            worst = min(factors.items(), key=lambda x: x[1].get('correlation', 999))
            print(f"  {horizon}M: Best={best[0]} (r={best[1]['correlation']:+.3f}), " +
                  f"Worst={worst[0]} (r={worst[1]['correlation']:+.3f})")

    print("\n" + "=" * 80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run S&P 500 grade validation test")
    parser.add_argument(
        "--yfinance",
        action="store_true",
        help="Use yfinance current fundamentals (has look-ahead bias)",
    )
    parser.add_argument(
        "--historical",
        action="store_true",
        help="Use SEC EDGAR historical fundamentals (no look-ahead bias, default)",
    )
    args = parser.parse_args()

    # Default to historical unless --yfinance specified
    use_historical = not args.yfinance

    results = run_validation(use_historical_fundamentals=use_historical)
