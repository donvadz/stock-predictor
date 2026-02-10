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
  const timerRef = useRef(null)

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

  const highConviction = result?.high_conviction || []
  const ultraConviction = result?.ultra_conviction || []

  return (
    <div className="card optimal-scan-card">
      <div className="card-header-badge">
        <span className="badge-recommended">Recommended</span>
      </div>
      <h2>High-Conviction Signals</h2>
      <p className="card-description">
        Our best strategy: finds stocks where <strong>4+ of 5 signals</strong> align bullish.
        Uses ML prediction + trend + volatility + momentum + RSI.
        <br />
        <span className="accuracy-highlight">Backtested: 78% accuracy on Tier 1 stocks (2023-2024)</span>
      </p>

      <div className="tier-stocks">
        <span className="tier-label">Tier 1 Stocks:</span>
        <span className="tier-list">MCD, QQQ, MSFT, MA, COST, SPY</span>
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
          <p>Analyzing {result?.optimal_stocks?.length || 12} optimal stocks...</p>
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
              <p className="section-note">All 5 signals aligned - highest confidence</p>
              <div className="signal-cards">
                {ultraConviction.map((signal) => (
                  <div key={signal.ticker} className="signal-card ultra">
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

          {/* High Conviction Signals */}
          {highConviction.length > 0 && (
            <div className="signals-section high-section">
              <h3>
                <span className="signal-icon">🎯</span>
                High Conviction (4+/5 Signals)
              </h3>
              <p className="section-note">{highConviction.length} signals found</p>
              <div className="signal-cards">
                {highConviction
                  .filter(s => !s.ultra_conviction)
                  .map((signal) => (
                    <div key={signal.ticker} className="signal-card high">
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
          {highConviction.length === 0 && (
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

          {/* Methodology */}
          <div className="methodology-box">
            <h4>Signal Methodology</h4>
            <ul>
              {result.methodology?.signals?.map((signal, i) => (
                <li key={i}>{signal}</li>
              ))}
            </ul>
            <p className="methodology-note">
              Backtested accuracy: {result.backtested_accuracy?.high_conviction_4_signals} (4+ signals),
              {' '}{result.backtested_accuracy?.ultra_conviction_5_signals} (5 signals)
            </p>
          </div>
        </div>
      )}
    </div>
  )
}

export default OptimalScan
