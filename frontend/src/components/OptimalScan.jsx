import { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function OptimalScan() {
  const [days, setDays] = useState(20)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [showTierInfo, setShowTierInfo] = useState(false)
  const timerRef = useRef(null)

  // Backtest state
  const [backtestLoading, setBacktestLoading] = useState(false)
  const [backtestResult, setBacktestResult] = useState(null)
  const [backtestError, setBacktestError] = useState(null)
  const [backtestElapsed, setBacktestElapsed] = useState(0)
  const [showBacktest, setShowBacktest] = useState(false)
  const backtestTimerRef = useRef(null)

  useEffect(() => {
    if (loading) {
      setElapsedSeconds(0)
      timerRef.current = setInterval(() => {
        setElapsedSeconds(s => s + 1)
      }, 1000)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [loading])

  // Backtest timer effect
  useEffect(() => {
    if (backtestLoading) {
      setBacktestElapsed(0)
      backtestTimerRef.current = setInterval(() => {
        setBacktestElapsed(s => s + 1)
      }, 1000)
    } else {
      if (backtestTimerRef.current) {
        clearInterval(backtestTimerRef.current)
        backtestTimerRef.current = null
      }
    }
    return () => {
      if (backtestTimerRef.current) clearInterval(backtestTimerRef.current)
    }
  }, [backtestLoading])

  const handleScan = async () => {
    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/optimal-scan?days=${days}`)
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Scan failed')
      }
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleBacktest = async () => {
    setBacktestLoading(true)
    setBacktestResult(null)
    setBacktestError(null)

    try {
      const response = await fetch(
        `${API_BASE}/optimal-backtest?days=${days}&min_signals=4&tier1_only=false`
      )
      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Backtest failed')
      }
      const data = await response.json()
      setBacktestResult(data)
    } catch (err) {
      setBacktestError(err.message)
    } finally {
      setBacktestLoading(false)
    }
  }

  const tier1Signals = result?.tier1_high_conviction || []
  const tier2Signals = result?.tier2_high_conviction || []
  const ultraConviction = result?.ultra_conviction || []
  const tierInfo = result?.tier_info || {}

  return (
    <div className="card optimal-scan-card">
      <div className="card-header-badge">
        <span className="badge-recommended">Recommended</span>
      </div>
      <h2>High-Conviction Signals</h2>
      <p className="card-description">
        Multi-signal consensus strategy: finds stocks where <strong>4+ of 5 signals</strong> align bullish.
        <br />
        <span className="accuracy-highlight">Tier 1: 80% accuracy | Tier 2: 65-75% accuracy (2023-2024)</span>
      </p>

      {/* Tier Info Toggle */}
      <div className="tier-stocks-container">
        <button
          className="tier-info-toggle"
          onClick={() => setShowTierInfo(!showTierInfo)}
        >
          {showTierInfo ? 'Hide' : 'Show'} Stock Tiers
        </button>

        {showTierInfo && (
          <div className="tier-info-panel">
            <div className="tier-box tier1-box">
              <div className="tier-header">
                <span className="tier-badge tier1">Tier 1</span>
                <span className="tier-accuracy">80% accuracy</span>
              </div>
              <div className="tier-stocks-list">
                {result?.tier_info?.tier1?.stocks?.join(', ') || 'MCD, QQQ, MSFT, MA, COST, SPY, XLK, JPM, CL'}
              </div>
              <div className="tier-action">
                <strong>Action:</strong> Full position size, highest confidence
              </div>
            </div>
            <div className="tier-box tier2-box">
              <div className="tier-header">
                <span className="tier-badge tier2">Tier 2</span>
                <span className="tier-accuracy">65-75% accuracy</span>
              </div>
              <div className="tier-stocks-list">
                {result?.tier_info?.tier2?.stocks?.join(', ') || 'AAPL, V, PG, HD, ICE, GS, ORLY, RSG, DUK, SRE, HON, CAT'}
              </div>
              <div className="tier-action">
                <strong>Action:</strong> Half position or wait for 5/5 signals
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="form-group">
        <label htmlFor="optimal-days">Holding Period</label>
        <div className="slider-container">
          <input
            id="optimal-days"
            type="range"
            min="5"
            max="30"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            disabled={loading}
          />
          <span className="slider-value">{days} days</span>
        </div>
        <span className="input-hint">20 days recommended (highest accuracy)</span>
      </div>

      <button onClick={handleScan} disabled={loading}>
        {loading && <span className="spinner"></span>}
        {loading ? 'Scanning...' : 'Scan for Signals'}
      </button>

      {loading && (
        <div className="loading-text">
          <p>Analyzing {result?.optimal_stocks?.length || 96} optimal stocks...</p>
          <p className="loading-timer">
            Elapsed: {formatTime(elapsedSeconds)}
          </p>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="optimal-results">
          {/* Regime Warning */}
          {result.regime_warning && (
            <div className="regime-warning-box">
              {result.regime_warning}
            </div>
          )}

          {/* Ultra Conviction Signals */}
          {ultraConviction.length > 0 && (
            <div className="signals-section ultra-section">
              <h3>
                <span className="signal-icon">⭐</span>
                Ultra Conviction (5/5 Signals)
              </h3>
              <p className="section-note">All 5 signals aligned - highest confidence, any tier</p>
              <div className="signal-cards">
                {ultraConviction.map((signal) => (
                  <div key={signal.ticker} className={`signal-card ultra tier${signal.tier}`}>
                    <div className="signal-tier-badge">Tier {signal.tier}</div>
                    <div className="signal-ticker">{signal.ticker}</div>
                    <div className="signal-price">${signal.price}</div>
                    <div className="signal-prediction">{signal.prediction}</div>
                    <div className="signal-confidence">{signal.confidence}% conf</div>
                    <div className="signal-details">
                      <span>RSI: {signal.rsi}</span>
                      <span>Vol: {signal.volatility}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tier 1 High Conviction Signals */}
          {tier1Signals.length > 0 && (
            <div className="signals-section tier1-section">
              <h3>
                <span className="signal-icon">🎯</span>
                Tier 1 Signals
                <span className="tier-accuracy-badge">80% accuracy</span>
              </h3>
              <p className="section-note">
                Best performers - full position size recommended
              </p>
              <div className="signal-cards">
                {tier1Signals
                  .filter(s => !s.ultra_conviction)
                  .map((signal) => (
                    <div key={signal.ticker} className="signal-card tier1">
                      <div className="signal-tier-badge tier1">Tier 1</div>
                      <div className="signal-ticker">{signal.ticker}</div>
                      <div className="signal-price">${signal.price}</div>
                      <div className="signal-prediction">{signal.prediction}</div>
                      <div className="signal-confidence">{signal.confidence}% conf</div>
                      <div className="signal-signals">{signal.signals_aligned}/5 signals</div>
                      <div className="signal-details">
                        <span>RSI: {signal.rsi}</span>
                        <span>Vol: {signal.volatility}%</span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Tier 2 High Conviction Signals */}
          {tier2Signals.length > 0 && (
            <div className="signals-section tier2-section">
              <h3>
                <span className="signal-icon">📊</span>
                Tier 2 Signals
                <span className="tier-accuracy-badge tier2">65-75% accuracy</span>
              </h3>
              <p className="section-note">
                Good performers - consider half position or require 5/5 signals
              </p>
              <div className="signal-cards">
                {tier2Signals
                  .filter(s => !s.ultra_conviction)
                  .map((signal) => (
                    <div key={signal.ticker} className="signal-card tier2">
                      <div className="signal-tier-badge tier2">Tier 2</div>
                      <div className="signal-ticker">{signal.ticker}</div>
                      <div className="signal-price">${signal.price}</div>
                      <div className="signal-prediction">{signal.prediction}</div>
                      <div className="signal-confidence">{signal.confidence}% conf</div>
                      <div className="signal-signals">{signal.signals_aligned}/5 signals</div>
                      <div className="signal-details">
                        <span>RSI: {signal.rsi}</span>
                        <span>Vol: {signal.volatility}%</span>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* No Signals */}
          {tier1Signals.length === 0 && tier2Signals.length === 0 && (
            <div className="no-signals">
              <span className="no-signals-icon">📭</span>
              <h4>No High-Conviction Signals Today</h4>
              <p>
                This is normal - signals occur 1-2 times per month.
                <br />
                Patience is part of the strategy.
              </p>
            </div>
          )}

          {/* Action Guide */}
          {(tier1Signals.length > 0 || tier2Signals.length > 0) && (
            <div className="action-guide">
              <h4>How to Use These Signals</h4>
              <div className="action-grid">
                <div className="action-item tier1-action">
                  <strong>Tier 1 (4+ signals)</strong>
                  <p>Full position size. These have 80% historical accuracy.</p>
                </div>
                <div className="action-item tier2-action">
                  <strong>Tier 2 (4 signals)</strong>
                  <p>Half position size, or wait for 5/5 signals before entering.</p>
                </div>
                <div className="action-item ultra-action">
                  <strong>Any Tier (5/5 signals)</strong>
                  <p>Highest conviction. Full position regardless of tier.</p>
                </div>
              </div>
            </div>
          )}

          {/* Methodology */}
          <div className="methodology-box">
            <h4>Signal Methodology</h4>
            <ul>
              <li><strong>ML prediction with 70%+ confidence (required)</strong></li>
              <li>Trend aligned (price {'>'} SMA20 {'>'} SMA50)</li>
              <li>Low volatility ({'<'} 35% annualized)</li>
              <li>Positive 5-day momentum</li>
              <li>RSI not overbought ({'<'} 70)</li>
            </ul>
            <p className="methodology-note">
              High conviction requires ML confidence 70%+ PLUS 3+ other signals aligned.
            </p>
          </div>
        </div>
      )}

      {/* Backtest Section */}
      <div className="backtest-section">
        <button
          className="backtest-toggle"
          onClick={() => setShowBacktest(!showBacktest)}
        >
          {showBacktest ? 'Hide' : 'Show'} Historical Backtest
        </button>

        {showBacktest && (
          <div className="backtest-panel">
            <h4>Realistic Backtest (2023-2024)</h4>
            <p className="backtest-description">
              Train on 2020-2022, test on 2023-2024 with no retraining.
              This shows honest, out-of-sample accuracy.
            </p>

            <button
              className="run-backtest-btn"
              onClick={handleBacktest}
              disabled={backtestLoading}
            >
              {backtestLoading && <span className="spinner"></span>}
              {backtestLoading ? 'Running Backtest...' : 'Run Backtest'}
            </button>

            {backtestLoading && (
              <div className="loading-text">
                <p>Testing {days}-day predictions across all optimal stocks...</p>
                <p className="loading-timer">Elapsed: {formatTime(backtestElapsed)}</p>
                <p className="loading-hint">This takes 2-4 minutes</p>
              </div>
            )}

            {backtestError && <div className="error">{backtestError}</div>}

            {backtestResult && (
              <div className="backtest-results">
                <div className="backtest-summary">
                  <div className="summary-stat primary">
                    <span className="stat-value">
                      {backtestResult.summary?.high_conviction_accuracy}%
                    </span>
                    <span className="stat-label">High Conviction Accuracy</span>
                  </div>
                  <div className="summary-stat">
                    <span className="stat-value">
                      {backtestResult.summary?.high_conviction_picks}
                    </span>
                    <span className="stat-label">Total Signals</span>
                  </div>
                  <div className="summary-stat">
                    <span className="stat-value">
                      {backtestResult.summary?.high_conviction_avg_return > 0 ? '+' : ''}
                      {backtestResult.summary?.high_conviction_avg_return}%
                    </span>
                    <span className="stat-label">Avg Return per Trade</span>
                  </div>
                  {backtestResult.summary?.ultra_conviction_picks > 0 && (
                    <div className="summary-stat ultra">
                      <span className="stat-value">
                        {backtestResult.summary?.ultra_conviction_accuracy}%
                      </span>
                      <span className="stat-label">
                        Ultra (5/5) Accuracy ({backtestResult.summary?.ultra_conviction_picks} trades)
                      </span>
                    </div>
                  )}
                </div>

                <div className="backtest-comparison">
                  <h5>Accuracy Improvement</h5>
                  <div className="comparison-bar">
                    <div className="bar-item baseline">
                      <span className="bar-label">Baseline (ML only)</span>
                      <span className="bar-value">{backtestResult.summary?.baseline_accuracy}%</span>
                    </div>
                    <div className="bar-item improved">
                      <span className="bar-label">With 4+ Signals</span>
                      <span className="bar-value">{backtestResult.summary?.high_conviction_accuracy}%</span>
                    </div>
                    <div className="bar-improvement">
                      +{backtestResult.improvement}% improvement
                    </div>
                  </div>
                </div>

                {/* Per Stock Breakdown */}
                {backtestResult.per_stock?.length > 0 && (
                  <div className="per-stock-breakdown">
                    <h5>Per-Stock Performance (Top 10)</h5>
                    <div className="stock-table">
                      <div className="stock-table-header">
                        <span>Stock</span>
                        <span>Accuracy</span>
                        <span>Signals</span>
                        <span>Avg Return</span>
                      </div>
                      {backtestResult.per_stock.slice(0, 10).map((stock) => (
                        <div key={stock.ticker} className="stock-table-row">
                          <span className="stock-ticker">{stock.ticker}</span>
                          <span className={`stock-accuracy ${stock.accuracy >= 70 ? 'high' : stock.accuracy >= 55 ? 'medium' : 'low'}`}>
                            {stock.accuracy}%
                          </span>
                          <span>{stock.picks}</span>
                          <span className={stock.avg_return >= 0 ? 'positive' : 'negative'}>
                            {stock.avg_return > 0 ? '+' : ''}{stock.avg_return}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="backtest-caveats">
                  <h5>Important Caveats</h5>
                  <ul>
                    {backtestResult.caveats?.map((caveat, i) => (
                      <li key={i}>{caveat}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default OptimalScan
