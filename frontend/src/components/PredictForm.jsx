import { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function PredictForm() {
  const [ticker, setTicker] = useState('')
  const [days, setDays] = useState(5)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!ticker.trim()) return

    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const response = await fetch(
        `${API_BASE}/predict?ticker=${encodeURIComponent(ticker)}&days=${days}`
      )

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <h2>Single Stock Prediction</h2>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="ticker">Stock Ticker</label>
          <input
            id="ticker"
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="e.g., AAPL"
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="days">Days Ahead (1-30)</label>
          <input
            id="days"
            type="number"
            min="1"
            max="30"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            disabled={loading}
          />
        </div>

        <button type="submit" disabled={loading || !ticker.trim()}>
          {loading && <span className="spinner"></span>}
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {loading && (
        <p className="loading-text">Analyzing {ticker}...</p>
      )}

      {error && (
        <div className="error">{error}</div>
      )}

      {result && (
        <div className={`result ${result.direction}`}>
          <div className={`result-direction ${result.direction}`}>
            {result.direction === 'up' ? '↑ UP' : '↓ DOWN'}
          </div>
          <div className="result-ticker">
            {result.ticker} • {result.days} day{result.days > 1 ? 's' : ''} ahead
          </div>
          <div className="result-confidence">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  )
}

export default PredictForm
