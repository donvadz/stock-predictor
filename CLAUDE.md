# Stock Direction Predictor

## Project Overview
A full-stack stock prediction application that uses machine learning to predict stock price direction (up/down) with confidence scores.

## Tech Stack
- **Backend**: FastAPI (Python 3.9), scikit-learn (RandomForest), pandas, numpy
- **Frontend**: React + Vite (JSX), located in `frontend/`
- **Data Source**: yfinance for OHLCV data
- **Deployment**: Railway (backend), Vercel (frontend)

## Project Structure
```
├── app.py                    # Main FastAPI application, all API endpoints
├── config.py                 # Configuration: stock lists, cache TTLs, model params
├── data.py                   # Data fetching from yfinance
├── cache.py                  # Caching layer
├── scoring.py                # SINGLE SOURCE OF TRUTH for all scoring logic
├── sec_edgar.py              # SEC EDGAR API client for historical fundamentals
├── historical_fundamentals.py # Point-in-time fundamental metrics from SEC data
├── model.py                  # Core ML prediction model
├── model_enhanced.py         # Enhanced prediction model
├── backtest.py               # Backtesting engine
├── backtest_realistic.py     # Realistic backtesting with slippage
├── composite_score.py        # Multi-factor composite scoring (uses scoring.py)
├── composite_backtest*.py    # Composite score backtesting variants
├── run_sp500_validation.py   # S&P 500 grade validation test runner
├── regime_aware.py           # Market regime detection (bull/bear/normal)
├── optimal_strategy.py       # Optimal strategy calculations
├── validation.py             # Model validation and feature importance
├── stress_test.py            # Crisis period testing
├── find_tier1_candidates.py  # Tier 1 stock candidate finder
├── job_manager.py            # Background job management
└── frontend/
    └── src/
        ├── App.jsx           # Main React app
        ├── App.css           # Styles
        └── components/       # React components
            ├── PredictForm.jsx
            ├── CompositeRanking.jsx
            ├── Backtest.jsx
            ├── Screener.jsx
            └── ...
```

## Key Concepts
- **Prediction Horizon**: 1-30 days, uses 1yr data for short-term (1-7d), 3yr for long-term (8+d)
- **Composite Score**: Multi-factor ranking combining ML predictions with market metrics
- **Regime Detection**: Bull/Bear/Normal based on SPY trailing returns
- **Backtesting**: Historical validation with realistic trading costs

## Common Commands
```bash
# Run backend
uvicorn app:app --reload --port 8000

# Run full stack
cd frontend && npm run dev:full

# Frontend only
cd frontend && npm run dev
```

## API Endpoints (in app.py)
- `GET /predict?ticker=AAPL&days=5` - Single stock prediction
- `GET /composite/rankings` - Composite score rankings
- `GET /backtest/*` - Various backtesting endpoints
- `GET /regime` - Current market regime
- `GET /screener/*` - Stock screening endpoints

## Stock Lists
Defined in `config.py`:
- `STOCK_LIST`: 100 stocks/ETFs including tech, finance, healthcare, Islamic ETFs
- `COMPOSITE_STOCK_LIST`: Extended list for composite scoring

## Environment Variables
- `FINNHUB_API_KEY` - For fundamental data
- `FRONTEND_URL` - For CORS (production)
- `VITE_API_URL` - Frontend API URL

## Coding Patterns
- Cache predictions with TTL (6 hours default)
- Use ThreadPoolExecutor for parallel stock processing
- Pydantic models for API responses
- Error handling with HTTPException

## Scoring System (Feb 2026 - Optimized)

The scoring system uses **optimized horizon-aware weights** based on SEC EDGAR backtesting:
- Tested 443 S&P 500 stocks with 85K+ validation samples
- Point-in-time fundamentals (no look-ahead bias)
- Grid search optimization over 20K+ weight combinations

### 1. Sector-Specific Valuation Thresholds
P/E ratios are scored relative to sector norms (defined in `scoring.py`):
- Tech: low=18, median=28, high=45
- Financials: low=8, median=12, high=18
- Utilities: low=14, median=18, high=24
- etc.

### 2. Sentiment/Contrarian Factor (Short-Term Only, ≤12M)
For horizons ≤12M, includes sentiment scoring based on:
- **Short Interest** (40%): High short + quality = squeeze potential
- **Analyst Sentiment** (30%): Contrarian signal when extreme
- **Earnings Surprise** (30%): Post-earnings drift momentum

### 3. OPTIMIZED Horizon-Aware Weight Profiles

| Horizon | Valuation | Sentiment | Quality | Financial | Growth |
|---------|-----------|-----------|---------|-----------|--------|
| 1-3M    | 21%       | 53%       | 0%      | 11%       | 16%    |
| 4-6M    | 26%       | 53%       | 0%      | 21%       | 0%     |
| 7-12M   | 30%       | 50%       | 0%      | 20%       | 0%     |
| 24M     | 43%       | 0%        | 0%      | 48%       | 10%    |
| 60M+    | 42%       | 0%        | 0%      | 47%       | 11%    |

### Key Findings from Optimization:
- **Valuation WORKS**: Positive correlation (+0.04 to +0.08) with returns
- **Quality = 0%**: Negative correlation (-0.04 to -0.07) - already priced in
- **Sentiment dominates short-term**: 50-53% weight for horizons ≤12M
- **Financial strength dominates long-term**: 47-48% weight for 24M+
- **Growth minimized**: Slight negative correlation, momentum doesn't persist

## Grade Validation Test Results (S&P 500, Optimized Weights)

Results with SEC EDGAR point-in-time fundamentals:

| Horizon | Previous A+B% | Optimized A+B% | Improvement |
|---------|---------------|----------------|-------------|
| 3M      | 57.4%         | **61.2%**      | +3.7%       |
| 6M      | 62.8%         | **66.1%**      | +3.3%       |
| 12M     | 52.7%         | **57.4%**      | +4.7%       |
| 24M     | 56.2%         | **59.0%**      | +2.7%       |
| 60M     | 64.4%         | **69.3%**      | +4.9%       |

### Running Validation & Optimization
```bash
python run_sp500_validation.py           # Full S&P 500 validation (SEC EDGAR)
python run_sp500_validation.py --yfinance # Use yfinance (has look-ahead bias)
python optimize_weights.py               # Re-run weight optimization
python optimize_weights.py --quick       # Quick test with 100 stocks
```
