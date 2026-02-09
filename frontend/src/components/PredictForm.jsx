import { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Stock list with names for searching
const STOCKS_WITH_NAMES = [
  { ticker: "AAPL", name: "Apple Inc." },
  { ticker: "MSFT", name: "Microsoft Corporation" },
  { ticker: "GOOGL", name: "Alphabet Inc." },
  { ticker: "AMZN", name: "Amazon.com Inc." },
  { ticker: "NVDA", name: "NVIDIA Corporation" },
  { ticker: "META", name: "Meta Platforms Inc." },
  { ticker: "TSLA", name: "Tesla Inc." },
  { ticker: "AMD", name: "Advanced Micro Devices" },
  { ticker: "INTC", name: "Intel Corporation" },
  { ticker: "CRM", name: "Salesforce Inc." },
  { ticker: "JPM", name: "JPMorgan Chase & Co." },
  { ticker: "V", name: "Visa Inc." },
  { ticker: "MA", name: "Mastercard Inc." },
  { ticker: "BAC", name: "Bank of America Corp." },
  { ticker: "WFC", name: "Wells Fargo & Co." },
  { ticker: "GS", name: "Goldman Sachs Group" },
  { ticker: "MS", name: "Morgan Stanley" },
  { ticker: "AXP", name: "American Express Co." },
  { ticker: "BLK", name: "BlackRock Inc." },
  { ticker: "C", name: "Citigroup Inc." },
  { ticker: "UNH", name: "UnitedHealth Group" },
  { ticker: "JNJ", name: "Johnson & Johnson" },
  { ticker: "PFE", name: "Pfizer Inc." },
  { ticker: "ABBV", name: "AbbVie Inc." },
  { ticker: "MRK", name: "Merck & Co." },
  { ticker: "LLY", name: "Eli Lilly and Co." },
  { ticker: "TMO", name: "Thermo Fisher Scientific" },
  { ticker: "ABT", name: "Abbott Laboratories" },
  { ticker: "DHR", name: "Danaher Corporation" },
  { ticker: "BMY", name: "Bristol-Myers Squibb" },
  { ticker: "WMT", name: "Walmart Inc." },
  { ticker: "PG", name: "Procter & Gamble Co." },
  { ticker: "KO", name: "Coca-Cola Company" },
  { ticker: "PEP", name: "PepsiCo Inc." },
  { ticker: "COST", name: "Costco Wholesale" },
  { ticker: "NKE", name: "Nike Inc." },
  { ticker: "MCD", name: "McDonald's Corporation" },
  { ticker: "SBUX", name: "Starbucks Corporation" },
  { ticker: "HD", name: "Home Depot Inc." },
  { ticker: "LOW", name: "Lowe's Companies" },
  { ticker: "XOM", name: "Exxon Mobil Corp." },
  { ticker: "CVX", name: "Chevron Corporation" },
  { ticker: "DIS", name: "Walt Disney Company" },
  { ticker: "NFLX", name: "Netflix Inc." },
  { ticker: "PYPL", name: "PayPal Holdings" },
  { ticker: "UBER", name: "Uber Technologies" },
  { ticker: "ABNB", name: "Airbnb Inc." },
  { ticker: "SQ", name: "Block Inc." },
  { ticker: "COIN", name: "Coinbase Global" },
  { ticker: "SNAP", name: "Snap Inc." },
  { ticker: "HLAL", name: "Wahed FTSE USA Shariah ETF" },
  { ticker: "SPUS", name: "SP Funds S&P 500 Sharia ETF" },
  { ticker: "SPRE", name: "SP Funds S&P Global REIT Sharia" },
  { ticker: "SPSK", name: "SP Funds Dow Jones Sukuk ETF" },
  { ticker: "UMMA", name: "Wahed Dow Jones Islamic World" },
  { ticker: "SPTE", name: "SP Funds S&P Global Tech Sharia" },
  { ticker: "ISDU.L", name: "iShares MSCI USA Islamic" },
  { ticker: "ISDE.L", name: "iShares MSCI EM Islamic" },
  { ticker: "ISWD.L", name: "iShares MSCI World Islamic" },
  { ticker: "IUSE.L", name: "iShares MSCI USA Islamic USD" },
  { ticker: "IQSA.L", name: "iShares MSCI Saudi Arabia" },
  { ticker: "HMWO.L", name: "HSBC MSCI World ETF" },
  { ticker: "HMEU.L", name: "HSBC MSCI Europe ETF" },
  { ticker: "HMUS.L", name: "HSBC MSCI USA ETF" },
  { ticker: "HMEM.L", name: "HSBC MSCI EM ETF" },
  { ticker: "HMJP.L", name: "HSBC MSCI Japan ETF" },
  { ticker: "SPY", name: "SPDR S&P 500 ETF" },
  { ticker: "QQQ", name: "Invesco QQQ Nasdaq 100" },
  { ticker: "IWM", name: "iShares Russell 2000 ETF" },
  { ticker: "DIA", name: "SPDR Dow Jones ETF" },
  { ticker: "VTI", name: "Vanguard Total Stock Market" },
  { ticker: "VOO", name: "Vanguard S&P 500 ETF" },
  { ticker: "XLK", name: "Technology Select Sector" },
  { ticker: "XLF", name: "Financial Select Sector" },
  { ticker: "XLV", name: "Healthcare Select Sector" },
  { ticker: "XLE", name: "Energy Select Sector" },
  { ticker: "XLI", name: "Industrial Select Sector" },
  { ticker: "XLY", name: "Consumer Discretionary Sector" },
  { ticker: "EFA", name: "iShares MSCI EAFE ETF" },
  { ticker: "EEM", name: "iShares MSCI Emerging Markets" },
  { ticker: "VWO", name: "Vanguard FTSE Emerging Markets" },
  { ticker: "FXI", name: "iShares China Large-Cap ETF" },
  { ticker: "EWJ", name: "iShares MSCI Japan ETF" },
  { ticker: "EWG", name: "iShares MSCI Germany ETF" },
  { ticker: "BND", name: "Vanguard Total Bond Market" },
  { ticker: "TLT", name: "iShares 20+ Year Treasury" },
  { ticker: "HYG", name: "iShares High Yield Corporate" },
  { ticker: "GLD", name: "SPDR Gold Shares" },
  { ticker: "SLV", name: "iShares Silver Trust" },
  { ticker: "USO", name: "United States Oil Fund" },
  { ticker: "ARKK", name: "ARK Innovation ETF" },
  { ticker: "ICLN", name: "iShares Global Clean Energy" },
  { ticker: "SOXX", name: "iShares Semiconductor ETF" },
  { ticker: "NET", name: "Cloudflare Inc." },
  { ticker: "CRWD", name: "CrowdStrike Holdings" },
  { ticker: "ZS", name: "Zscaler Inc." },
  { ticker: "S", name: "SentinelOne Inc." },
  { ticker: "DDOG", name: "Datadog Inc." },
  { ticker: "SNOW", name: "Snowflake Inc." },
  { ticker: "MDB", name: "MongoDB Inc." },
  { ticker: "ESTC", name: "Elastic N.V." },
  { ticker: "CFLT", name: "Confluent Inc." },
  { ticker: "GTLB", name: "GitLab Inc." },
  { ticker: "PLTR", name: "Palantir Technologies" },
  { ticker: "AI", name: "C3.ai Inc." },
  { ticker: "PATH", name: "UiPath Inc." },
  { ticker: "HUBS", name: "HubSpot Inc." },
  { ticker: "TTD", name: "The Trade Desk Inc." },
  { ticker: "U", name: "Unity Software Inc." },
  { ticker: "RBLX", name: "Roblox Corporation" },
  { ticker: "DUOL", name: "Duolingo Inc." },
  { ticker: "ASAN", name: "Asana Inc." },
  { ticker: "DOCN", name: "DigitalOcean Holdings" },
  { ticker: "RIVN", name: "Rivian Automotive" },
  { ticker: "LCID", name: "Lucid Group Inc." },
  { ticker: "NIO", name: "NIO Inc." },
  { ticker: "XPEV", name: "XPeng Inc." },
  { ticker: "LI", name: "Li Auto Inc." },
  { ticker: "CHPT", name: "ChargePoint Holdings" },
  { ticker: "PLUG", name: "Plug Power Inc." },
  { ticker: "FSLR", name: "First Solar Inc." },
  { ticker: "ENPH", name: "Enphase Energy Inc." },
  { ticker: "SEDG", name: "SolarEdge Technologies" },
  { ticker: "RUN", name: "Sunrun Inc." },
  { ticker: "BE", name: "Bloom Energy Corp." },
  { ticker: "STEM", name: "Stem Inc." },
  { ticker: "MRVL", name: "Marvell Technology" },
  { ticker: "ON", name: "ON Semiconductor" },
  { ticker: "WOLF", name: "Wolfspeed Inc." },
  { ticker: "LSCC", name: "Lattice Semiconductor" },
  { ticker: "CRUS", name: "Cirrus Logic Inc." },
  { ticker: "SITM", name: "SiTime Corporation" },
  { ticker: "DXCM", name: "DexCom Inc." },
  { ticker: "ISRG", name: "Intuitive Surgical" },
  { ticker: "VEEV", name: "Veeva Systems Inc." },
  { ticker: "DOCS", name: "Doximity Inc." },
  { ticker: "HIMS", name: "Hims & Hers Health" },
  { ticker: "SHOP", name: "Shopify Inc." },
  { ticker: "ETSY", name: "Etsy Inc." },
  { ticker: "CHWY", name: "Chewy Inc." },
  { ticker: "SE", name: "Sea Limited" },
  { ticker: "MELI", name: "MercadoLibre Inc." },
  { ticker: "RKLB", name: "Rocket Lab USA" },
  { ticker: "ASTS", name: "AST SpaceMobile" },
  { ticker: "IRDM", name: "Iridium Communications" },
  { ticker: "IONQ", name: "IonQ Inc." },
]

function PredictForm() {
  const [ticker, setTicker] = useState('')
  const [searchText, setSearchText] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [days, setDays] = useState('5')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const dropdownRef = useRef(null)

  const daysNum = Number(days)
  const isValidDays = days !== '' && daysNum >= 1 && daysNum <= 30

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setShowSuggestions(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Filter suggestions based on input - search both ticker and name
  const filteredSuggestions = searchText.length > 0
    ? STOCKS_WITH_NAMES.filter(s =>
        s.ticker.toLowerCase().includes(searchText.toLowerCase()) ||
        s.name.toLowerCase().includes(searchText.toLowerCase())
      )
    : []

  const handleSelectStock = (stock) => {
    setTicker(stock.ticker)
    setSearchText(`${stock.ticker} - ${stock.name}`)
    setShowSuggestions(false)
  }

  const handleInputChange = (e) => {
    const value = e.target.value
    setSearchText(value)
    // If it looks like a ticker (all caps, short), use it directly
    const upperValue = value.toUpperCase().trim()
    if (upperValue.length <= 6 && /^[A-Z.]+$/.test(upperValue)) {
      setTicker(upperValue)
    } else {
      setTicker('')
    }
    setShowSuggestions(value.length > 0)
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!ticker.trim() || !isValidDays) return

    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const response = await fetch(
        `${API_BASE}/predict?ticker=${encodeURIComponent(ticker)}&days=${daysNum}`
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
      <p className="card-description">
        Predict whether a stock will go up or down within your chosen timeframe.
        Returns the predicted direction, confidence level, and expected price target.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="form-group" ref={dropdownRef}>
          <label htmlFor="ticker">Stock Ticker</label>
          <div className="dropdown-container">
            <input
              id="ticker"
              type="text"
              value={searchText}
              onChange={handleInputChange}
              onFocus={() => searchText.length > 0 && setShowSuggestions(true)}
              placeholder="Search by ticker or company name..."
              disabled={loading}
              autoComplete="off"
            />
            {showSuggestions && filteredSuggestions.length > 0 && (
              <div className="dropdown-list">
                {filteredSuggestions.slice(0, 8).map(stock => (
                  <div
                    key={stock.ticker}
                    className="dropdown-item-with-name"
                    onClick={() => handleSelectStock(stock)}
                  >
                    <span className="dropdown-ticker">{stock.ticker}</span>
                    <span className="dropdown-name">{stock.name}</span>
                  </div>
                ))}
                {filteredSuggestions.length > 8 && (
                  <div className="dropdown-more">
                    +{filteredSuggestions.length - 8} more
                  </div>
                )}
              </div>
            )}
          </div>
          <small className="input-hint">Search from our list, or enter any Yahoo Finance ticker directly</small>
        </div>

        <div className="form-group">
          <label htmlFor="days">Days Ahead (1-30)</label>
          <input
            id="days"
            type="number"
            min="1"
            max="30"
            value={days}
            onChange={(e) => setDays(e.target.value)}
            disabled={loading}
          />
        </div>

        <button type="submit" disabled={loading || !ticker.trim() || !isValidDays}>
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
          <div className="result-name">
            {result.name}
          </div>
          <div className="result-ticker">
            {result.ticker} • {result.days} day{result.days > 1 ? 's' : ''} ahead
          </div>
          <div className="result-price">
            ${result.latest_price} → ${result.predicted_price}
            <span className={result.predicted_change >= 0 ? 'change-up' : 'change-down'}>
              {' '}({result.predicted_change >= 0 ? '+' : ''}{result.predicted_change}%)
            </span>
          </div>
          <div className="result-confidence">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </div>

          {/* Market Regime Section */}
          {result.market_regime && (
            <div className="prediction-regime">
              <h4>Market Conditions</h4>
              <div className="regime-status">
                <span className={`regime-status-badge ${result.market_regime.regime}`}>
                  {result.market_regime.regime === 'bear' ? '🐻' :
                   result.market_regime.regime === 'bull' ? '🐂' : '📊'}
                  {' '}{result.market_regime.regime.toUpperCase()}
                </span>
                <span className="regime-spy-small">
                  SPY: {result.market_regime.spy_return >= 0 ? '+' : ''}{result.market_regime.spy_return}%
                </span>
              </div>
              <div className="regime-threshold-info">
                <span className={`regime-threshold-status ${result.market_regime.meets_threshold ? 'meets' : 'below'}`}>
                  {result.market_regime.meets_threshold
                    ? '✓ Meets regime threshold'
                    : `⚠ Below ${(result.market_regime.recommended_confidence * 100).toFixed(0)}% threshold for ${result.market_regime.regime} market`
                  }
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default PredictForm
