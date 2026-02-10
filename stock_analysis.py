"""
Stock Characteristics vs Prediction Accuracy Analysis

PURPOSE:
This module analyzes why certain stocks (MCD 64%, HD 58%, QQQ 58%, MSFT 58%)
performed much better in out-of-sample testing than others (TSLA 41%, LLY 45%, GOOGL 46%).

ANALYSIS:
1. Fetches characteristics for each stock (volatility, beta, market cap, sector, volume)
2. Correlates these characteristics with out-of-sample accuracy results
3. Identifies patterns - do low-volatility, defensive stocks predict better?
4. Outputs a clear report showing which stock characteristics correlate with predictability
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats


# =============================================================================
# VALIDATION RESULTS (from year-split backtest)
# =============================================================================

# Out-of-sample accuracy results from validation.py
VALIDATION_RESULTS = {
    # Top performers (>= 57%)
    "MCD": 64,
    "HD": 58,
    "QQQ": 58,
    "MSFT": 58,
    "AAPL": 57,
    "RIVN": 57,
    "MA": 57,

    # Mid performers (55-56%)
    "SPY": 56,
    "JNJ": 56,
    "BAC": 56,
    "KO": 55,
    "V": 55,

    # Near random (50-54%)
    "META": 52,
    "NVDA": 50,
    "AMD": 50,

    # Below random (< 50%)
    "JPM": 48,
    "UNH": 48,
    "AMZN": 48,
    "WMT": 47,
    "COIN": 47,
    "GOOGL": 46,
    "PFE": 46,
    "NKE": 46,
    "LLY": 45,
    "TSLA": 41,
}

# Sector mapping
SECTOR_NAMES = {
    "Technology": "Technology",
    "Healthcare": "Healthcare",
    "Financial Services": "Financial",
    "Consumer Cyclical": "Consumer Cyclical",
    "Communication Services": "Communication",
    "Consumer Defensive": "Consumer Defensive",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Basic Materials": "Basic Materials",
}


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_stock_characteristics(ticker: str) -> Optional[Dict]:
    """
    Fetch comprehensive characteristics for a stock.

    Returns:
        Dict with volatility, beta, market_cap, sector, avg_volume, and more
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Fetch historical data for volatility calculation
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)  # 2 years
        hist = stock.history(start=start_date, end=end_date, interval="1d")

        if hist.empty or len(hist) < 200:
            print(f"  Warning: Insufficient historical data for {ticker}")
            return None

        # Calculate daily returns
        returns = hist["Close"].pct_change().dropna()

        # Calculate volatility metrics
        volatility_daily = returns.std()
        volatility_annual = volatility_daily * np.sqrt(252)

        # Calculate average true range (ATR) as alternative volatility measure
        high_low = hist["High"] - hist["Low"]
        high_close = np.abs(hist["High"] - hist["Close"].shift())
        low_close = np.abs(hist["Low"] - hist["Close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_pct = (tr.rolling(14).mean() / hist["Close"]).iloc[-1]

        # Volume characteristics
        avg_volume = hist["Volume"].mean()
        volume_volatility = hist["Volume"].std() / avg_volume if avg_volume > 0 else 0

        # Price momentum characteristics
        price_52w_high = hist["High"].max()
        price_52w_low = hist["Low"].min()
        current_price = hist["Close"].iloc[-1]
        distance_from_52w_high = (price_52w_high - current_price) / price_52w_high

        # Return distribution characteristics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "ticker": ticker,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "market_cap_log": np.log10(info.get("marketCap", 1)) if info.get("marketCap", 0) > 0 else 0,
            "beta": info.get("beta", 1.0),
            "volatility_daily": volatility_daily,
            "volatility_annual": volatility_annual,
            "atr_pct": atr_pct,
            "avg_volume": avg_volume,
            "avg_dollar_volume": avg_volume * current_price,
            "volume_volatility": volume_volatility,
            "price": current_price,
            "distance_from_52w_high": distance_from_52w_high,
            "return_skewness": skewness,
            "return_kurtosis": kurtosis,
            "max_drawdown": max_drawdown,
            "dividend_yield": info.get("dividendYield", 0) or 0,
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "analyst_rating": info.get("recommendationMean"),  # 1=Buy, 5=Sell
            "short_percent": info.get("shortPercentOfFloat", 0) or 0,
            "is_etf": info.get("quoteType") == "ETF",
        }

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def classify_stock_type(characteristics: Dict) -> str:
    """Classify stock into categories based on characteristics."""
    sector = characteristics.get("sector", "Unknown")
    beta = characteristics.get("beta", 1.0) or 1.0
    volatility = characteristics.get("volatility_annual", 0.3)
    dividend_yield = characteristics.get("dividend_yield", 0) or 0
    is_etf = characteristics.get("is_etf", False)

    if is_etf:
        return "ETF"
    elif sector in ["Consumer Defensive", "Utilities"] or (dividend_yield > 0.02 and beta < 1.0):
        return "Defensive"
    elif sector == "Technology" and volatility > 0.4:
        return "High-Growth Tech"
    elif sector == "Technology":
        return "Stable Tech"
    elif sector == "Healthcare" and volatility > 0.35:
        return "Volatile Healthcare"
    elif sector == "Healthcare":
        return "Stable Healthcare"
    elif sector == "Financial Services":
        return "Financial"
    elif sector == "Consumer Cyclical":
        return "Consumer Cyclical"
    elif beta > 1.3 or volatility > 0.45:
        return "High Volatility"
    else:
        return "Other"


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def compute_correlations(df: pd.DataFrame, target_col: str = "accuracy") -> pd.DataFrame:
    """
    Compute correlations between stock characteristics and prediction accuracy.

    Returns DataFrame with correlation coefficients and p-values.
    """
    results = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col and not df[c].isna().all()]

    for col in numeric_cols:
        # Remove NaN values for this column
        valid_mask = ~df[col].isna() & ~df[target_col].isna()
        if valid_mask.sum() < 5:
            continue

        x = df.loc[valid_mask, col].values
        y = df.loc[valid_mask, target_col].values

        # Pearson correlation
        if len(set(x)) > 1:  # Need variation
            corr, p_value = stats.pearsonr(x, y)

            # Spearman (rank) correlation - more robust
            spearman_corr, spearman_p = stats.spearmanr(x, y)

            results.append({
                "characteristic": col,
                "pearson_corr": corr,
                "pearson_p": p_value,
                "spearman_corr": spearman_corr,
                "spearman_p": spearman_p,
                "n_samples": valid_mask.sum(),
                "significant": p_value < 0.1,  # 90% confidence
            })

    return pd.DataFrame(results).sort_values("pearson_corr", ascending=False)


def analyze_by_group(df: pd.DataFrame, group_col: str, target_col: str = "accuracy") -> pd.DataFrame:
    """Analyze accuracy by categorical grouping."""
    grouped = df.groupby(group_col)[target_col].agg(["mean", "std", "count"]).round(1)
    grouped.columns = ["avg_accuracy", "std_accuracy", "count"]
    return grouped.sort_values("avg_accuracy", ascending=False)


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(df: pd.DataFrame, correlations: pd.DataFrame) -> str:
    """Generate a comprehensive analysis report."""

    report = []
    report.append("=" * 80)
    report.append("STOCK PREDICTABILITY ANALYSIS REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    # Summary statistics
    report.append("\n" + "=" * 80)
    report.append("1. SUMMARY STATISTICS")
    report.append("=" * 80)
    report.append(f"Total stocks analyzed: {len(df)}")
    report.append(f"Average accuracy: {df['accuracy'].mean():.1f}%")
    report.append(f"Accuracy range: {df['accuracy'].min():.0f}% to {df['accuracy'].max():.0f}%")
    report.append(f"Accuracy std dev: {df['accuracy'].std():.1f}%")

    # Top and bottom performers
    report.append("\n" + "-" * 40)
    report.append("Top 5 Performers:")
    for _, row in df.nlargest(5, "accuracy").iterrows():
        report.append(f"  {row['ticker']:6s} {row['accuracy']:5.0f}%  ({row['sector']}, vol={row['volatility_annual']*100:.0f}%)")

    report.append("\nBottom 5 Performers:")
    for _, row in df.nsmallest(5, "accuracy").iterrows():
        report.append(f"  {row['ticker']:6s} {row['accuracy']:5.0f}%  ({row['sector']}, vol={row['volatility_annual']*100:.0f}%)")

    # Correlation analysis
    report.append("\n" + "=" * 80)
    report.append("2. CORRELATION ANALYSIS")
    report.append("=" * 80)
    report.append("\nCharacteristics vs Prediction Accuracy (sorted by correlation):")
    report.append("-" * 70)
    report.append(f"{'Characteristic':<25} {'Pearson r':>10} {'p-value':>10} {'Spearman r':>10} {'Signif':>8}")
    report.append("-" * 70)

    for _, row in correlations.iterrows():
        sig = "*" if row["significant"] else ""
        report.append(
            f"{row['characteristic']:<25} {row['pearson_corr']:>10.3f} {row['pearson_p']:>10.3f} "
            f"{row['spearman_corr']:>10.3f} {sig:>8}"
        )

    # Key findings
    report.append("\n" + "=" * 80)
    report.append("3. KEY FINDINGS")
    report.append("=" * 80)

    # Volatility analysis
    vol_corr = correlations[correlations["characteristic"] == "volatility_annual"]
    if not vol_corr.empty:
        vol_r = vol_corr.iloc[0]["pearson_corr"]
        if vol_r < -0.2:
            report.append(f"\n[VOLATILITY] STRONG NEGATIVE correlation (r={vol_r:.3f})")
            report.append("  -> LOW volatility stocks are MORE predictable")
            report.append("  -> This supports the hypothesis that stable, defensive stocks predict better")
        elif vol_r > 0.2:
            report.append(f"\n[VOLATILITY] POSITIVE correlation (r={vol_r:.3f})")
            report.append("  -> Surprisingly, higher volatility may help predictions")
        else:
            report.append(f"\n[VOLATILITY] WEAK correlation (r={vol_r:.3f})")
            report.append("  -> Volatility alone doesn't explain predictability differences")

    # Beta analysis
    beta_corr = correlations[correlations["characteristic"] == "beta"]
    if not beta_corr.empty:
        beta_r = beta_corr.iloc[0]["pearson_corr"]
        if beta_r < -0.2:
            report.append(f"\n[BETA] NEGATIVE correlation (r={beta_r:.3f})")
            report.append("  -> Low-beta (defensive) stocks predict better")
        elif beta_r > 0.2:
            report.append(f"\n[BETA] POSITIVE correlation (r={beta_r:.3f})")
            report.append("  -> High-beta stocks may be easier to predict")
        else:
            report.append(f"\n[BETA] WEAK correlation (r={beta_r:.3f})")

    # Sector analysis
    report.append("\n" + "=" * 80)
    report.append("4. SECTOR ANALYSIS")
    report.append("=" * 80)

    sector_stats = analyze_by_group(df, "sector")
    report.append(f"\n{'Sector':<25} {'Avg Acc':>10} {'Std':>8} {'Count':>8}")
    report.append("-" * 55)
    for sector, row in sector_stats.iterrows():
        report.append(f"{sector:<25} {row['avg_accuracy']:>9.1f}% {row['std_accuracy']:>7.1f} {row['count']:>8.0f}")

    # Stock type analysis
    if "stock_type" in df.columns:
        report.append("\n" + "=" * 80)
        report.append("5. STOCK TYPE ANALYSIS")
        report.append("=" * 80)

        type_stats = analyze_by_group(df, "stock_type")
        report.append(f"\n{'Stock Type':<25} {'Avg Acc':>10} {'Std':>8} {'Count':>8}")
        report.append("-" * 55)
        for stype, row in type_stats.iterrows():
            report.append(f"{stype:<25} {row['avg_accuracy']:>9.1f}% {row['std_accuracy']:>7.1f} {row['count']:>8.0f}")

    # Quartile analysis
    report.append("\n" + "=" * 80)
    report.append("6. VOLATILITY QUARTILE ANALYSIS")
    report.append("=" * 80)

    df["vol_quartile"] = pd.qcut(df["volatility_annual"], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
    vol_quartile_stats = analyze_by_group(df, "vol_quartile")
    report.append(f"\n{'Volatility Quartile':<25} {'Avg Acc':>10} {'Std':>8} {'Count':>8}")
    report.append("-" * 55)
    for q, row in vol_quartile_stats.iterrows():
        report.append(f"{q:<25} {row['avg_accuracy']:>9.1f}% {row['std_accuracy']:>7.1f} {row['count']:>8.0f}")

    # Detailed stock breakdown
    report.append("\n" + "=" * 80)
    report.append("7. DETAILED STOCK BREAKDOWN")
    report.append("=" * 80)
    report.append(f"\n{'Ticker':<6} {'Acc':>5} {'Sector':<20} {'Vol%':>6} {'Beta':>6} {'Type':<18}")
    report.append("-" * 75)

    for _, row in df.sort_values("accuracy", ascending=False).iterrows():
        beta_str = f"{row['beta']:.2f}" if pd.notna(row['beta']) else "N/A"
        report.append(
            f"{row['ticker']:<6} {row['accuracy']:>4.0f}% {row['sector']:<20} "
            f"{row['volatility_annual']*100:>5.0f}% {beta_str:>6} {row['stock_type']:<18}"
        )

    # Conclusions
    report.append("\n" + "=" * 80)
    report.append("8. CONCLUSIONS & RECOMMENDATIONS")
    report.append("=" * 80)

    # Determine main findings
    best_performers = df[df["accuracy"] >= 56]
    worst_performers = df[df["accuracy"] <= 48]

    if len(best_performers) > 0 and len(worst_performers) > 0:
        best_avg_vol = best_performers["volatility_annual"].mean()
        worst_avg_vol = worst_performers["volatility_annual"].mean()

        best_avg_beta = best_performers["beta"].mean()
        worst_avg_beta = worst_performers["beta"].mean()

        report.append(f"\nBest performers (>= 56% accuracy):")
        report.append(f"  - Average volatility: {best_avg_vol*100:.1f}%")
        report.append(f"  - Average beta: {best_avg_beta:.2f}")

        report.append(f"\nWorst performers (<= 48% accuracy):")
        report.append(f"  - Average volatility: {worst_avg_vol*100:.1f}%")
        report.append(f"  - Average beta: {worst_avg_beta:.2f}")

        if best_avg_vol < worst_avg_vol:
            report.append("\n[FINDING] Lower volatility stocks ARE more predictable")
        else:
            report.append("\n[FINDING] Volatility difference is not the main factor")

        if best_avg_beta < worst_avg_beta:
            report.append("[FINDING] Lower beta (defensive) stocks ARE more predictable")
        else:
            report.append("[FINDING] Beta is not a strong predictor of accuracy")

    # Sector insights
    best_sectors = sector_stats[sector_stats["avg_accuracy"] >= 55].index.tolist()
    worst_sectors = sector_stats[sector_stats["avg_accuracy"] <= 50].index.tolist()

    if best_sectors:
        report.append(f"\n[SECTOR INSIGHT] Best predicting sectors: {', '.join(best_sectors)}")
    if worst_sectors:
        report.append(f"[SECTOR INSIGHT] Worst predicting sectors: {', '.join(worst_sectors)}")

    # Final recommendations
    report.append("\n" + "-" * 40)
    report.append("RECOMMENDATIONS:")
    report.append("-" * 40)
    report.append("1. Focus model predictions on low-volatility, defensive stocks")
    report.append("2. Apply higher confidence thresholds to volatile/high-growth stocks")
    report.append("3. Consider sector-specific models for improved accuracy")
    report.append("4. ETFs (SPY, QQQ) show moderate predictability - good baseline")
    report.append("5. Be cautious with model predictions for TSLA, COIN, LLY, GOOGL")

    return "\n".join(report)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis(verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Run the full stock predictability analysis.

    Returns:
        Tuple of (characteristics_df, correlations_df, report_text)
    """
    print("=" * 70)
    print("STOCK PREDICTABILITY ANALYSIS")
    print(f"Analyzing {len(VALIDATION_RESULTS)} stocks")
    print("=" * 70)

    # Fetch characteristics for all stocks
    all_characteristics = []

    for ticker, accuracy in VALIDATION_RESULTS.items():
        print(f"\nFetching data for {ticker}...")
        chars = fetch_stock_characteristics(ticker)

        if chars:
            chars["accuracy"] = accuracy
            chars["stock_type"] = classify_stock_type(chars)
            all_characteristics.append(chars)
            print(f"  Volatility: {chars['volatility_annual']*100:.1f}%, Beta: {chars['beta']:.2f}, Sector: {chars['sector']}")
        else:
            print(f"  Skipped: Could not fetch data")

    # Create DataFrame
    df = pd.DataFrame(all_characteristics)

    print(f"\n{'=' * 70}")
    print(f"Successfully fetched data for {len(df)} stocks")
    print("=" * 70)

    # Compute correlations
    print("\nComputing correlations...")
    correlations = compute_correlations(df)

    # Generate report
    print("Generating report...")
    report = generate_report(df, correlations)

    if verbose:
        print("\n" + report)

    return df, correlations, report


def main():
    """Main entry point."""
    df, correlations, report = run_analysis(verbose=True)

    # Save results
    df.to_csv("/Users/aadilvadva/Code/stock-predictor/stock_characteristics.csv", index=False)
    correlations.to_csv("/Users/aadilvadva/Code/stock-predictor/characteristic_correlations.csv", index=False)

    with open("/Users/aadilvadva/Code/stock-predictor/predictability_report.txt", "w") as f:
        f.write(report)

    print("\n" + "=" * 70)
    print("FILES SAVED:")
    print("  - stock_characteristics.csv")
    print("  - characteristic_correlations.csv")
    print("  - predictability_report.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
