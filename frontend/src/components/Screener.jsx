import { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function Screener() {
  const [days, setDays] = useState(5)
  const [minConfidence, setMinConfidence] = useState(75)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()

    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const response = await fetch(
        `${API_BASE}/screener?days=${days}&min_confidence=${minConfidence / 100}`
      )

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.detail || 'Screener failed')
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
      <h2>Bullish Stock Screener</h2>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="screener-days">Days Ahead (1-30)</label>
          <input
            id="screener-days"
            type="number"
            min="1"
            max="30"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="min-confidence">Minimum Confidence</label>
          <div className="slider-container">
            <input
              id="min-confidence"
              type="range"
              min="50"
              max="100"
              value={minConfidence}
              onChange={(e) => setMinConfidence(Number(e.target.value))}
              disabled={loading}
            />
            <span className="slider-value">{minConfidence}%</span>
          </div>
        </div>

        <button type="submit" disabled={loading}>
          {loading && <span className="spinner"></span>}
          {loading ? 'Scanning...' : 'Scan Stocks'}
        </button>
      </form>

      {loading && (
        <p className="loading-text">Scanning 50 stocks... This may take ~30 seconds.</p>
      )}

      {error && (
        <div className="error">{error}</div>
      )}

      {result && (
        <>
          <div className="results-summary">
            Scanned {result.stocks_scanned} stocks • Found {result.matches.length} bullish
            {result.errors > 0 && ` • ${result.errors} errors`}
          </div>

          {result.matches.length > 0 ? (
            <div className="results-table">
              <table>
                <thead>
                  <tr>
                    <th>Ticker</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {result.matches.map((stock) => (
                    <tr key={stock.ticker}>
                      <td><strong>{stock.ticker}</strong></td>
                      <td>
                        <div className="confidence-bar">
                          <div
                            className="confidence-bar-fill"
                            style={{ width: `${stock.confidence * 100}px` }}
                          ></div>
                          <span className="confidence-value">
                            {(stock.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="no-results">
              No stocks found matching the criteria.
            </div>
          )}
        </>
      )}
    </div>
  )
}

export default Screener
