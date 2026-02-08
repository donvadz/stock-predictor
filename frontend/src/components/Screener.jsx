import { useState, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Stock categories for display with names
const STOCK_CATEGORIES = {
  "Tech Giants": [
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
  ],
  "Finance": [
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
  ],
  "Healthcare": [
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
  ],
  "Consumer": [
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
  ],
  "Other Major": [
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
  ],
  "Islamic ETFs (US)": [
    { ticker: "HLAL", name: "Wahed FTSE USA Shariah ETF" },
    { ticker: "SPUS", name: "SP Funds S&P 500 Sharia ETF" },
    { ticker: "SPRE", name: "SP Funds S&P Global REIT Sharia" },
    { ticker: "SPSK", name: "SP Funds Dow Jones Sukuk ETF" },
    { ticker: "UMMA", name: "Wahed Dow Jones Islamic World" },
    { ticker: "SPTE", name: "SP Funds S&P Global Tech Sharia" },
  ],
  "Islamic ETFs (London)": [
    { ticker: "ISDU.L", name: "iShares MSCI USA Islamic" },
    { ticker: "ISDE.L", name: "iShares MSCI EM Islamic" },
    { ticker: "ISWD.L", name: "iShares MSCI World Islamic" },
    { ticker: "IUSE.L", name: "iShares MSCI USA Islamic USD" },
    { ticker: "IQSA.L", name: "iShares MSCI Saudi Arabia" },
  ],
  "HSBC ETFs": [
    { ticker: "HMWO.L", name: "HSBC MSCI World ETF" },
    { ticker: "HMEU.L", name: "HSBC MSCI Europe ETF" },
    { ticker: "HMUS.L", name: "HSBC MSCI USA ETF" },
    { ticker: "HMEM.L", name: "HSBC MSCI EM ETF" },
    { ticker: "HMJP.L", name: "HSBC MSCI Japan ETF" },
  ],
  "Index ETFs": [
    { ticker: "SPY", name: "SPDR S&P 500 ETF" },
    { ticker: "QQQ", name: "Invesco QQQ Nasdaq 100" },
    { ticker: "IWM", name: "iShares Russell 2000 ETF" },
    { ticker: "DIA", name: "SPDR Dow Jones ETF" },
    { ticker: "VTI", name: "Vanguard Total Stock Market" },
    { ticker: "VOO", name: "Vanguard S&P 500 ETF" },
  ],
  "Sector ETFs": [
    { ticker: "XLK", name: "Technology Select Sector" },
    { ticker: "XLF", name: "Financial Select Sector" },
    { ticker: "XLV", name: "Healthcare Select Sector" },
    { ticker: "XLE", name: "Energy Select Sector" },
    { ticker: "XLI", name: "Industrial Select Sector" },
    { ticker: "XLY", name: "Consumer Discretionary Sector" },
  ],
  "International ETFs": [
    { ticker: "EFA", name: "iShares MSCI EAFE ETF" },
    { ticker: "EEM", name: "iShares MSCI Emerging Markets" },
    { ticker: "VWO", name: "Vanguard FTSE Emerging Markets" },
    { ticker: "FXI", name: "iShares China Large-Cap ETF" },
    { ticker: "EWJ", name: "iShares MSCI Japan ETF" },
    { ticker: "EWG", name: "iShares MSCI Germany ETF" },
  ],
  "Bond ETFs": [
    { ticker: "BND", name: "Vanguard Total Bond Market" },
    { ticker: "TLT", name: "iShares 20+ Year Treasury" },
    { ticker: "HYG", name: "iShares High Yield Corporate" },
  ],
  "Commodity ETFs": [
    { ticker: "GLD", name: "SPDR Gold Shares" },
    { ticker: "SLV", name: "iShares Silver Trust" },
    { ticker: "USO", name: "United States Oil Fund" },
  ],
  "Thematic ETFs": [
    { ticker: "ARKK", name: "ARK Innovation ETF" },
    { ticker: "ICLN", name: "iShares Global Clean Energy" },
    { ticker: "SOXX", name: "iShares Semiconductor ETF" },
  ],
  "Cloud & Cybersecurity": [
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
  ],
  "AI & Software": [
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
  ],
  "EVs & Clean Energy": [
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
  ],
  "Semiconductors": [
    { ticker: "MRVL", name: "Marvell Technology" },
    { ticker: "ON", name: "ON Semiconductor" },
    { ticker: "WOLF", name: "Wolfspeed Inc." },
    { ticker: "LSCC", name: "Lattice Semiconductor" },
    { ticker: "CRUS", name: "Cirrus Logic Inc." },
    { ticker: "SITM", name: "SiTime Corporation" },
  ],
  "Healthcare Tech": [
    { ticker: "DXCM", name: "DexCom Inc." },
    { ticker: "ISRG", name: "Intuitive Surgical" },
    { ticker: "VEEV", name: "Veeva Systems Inc." },
    { ticker: "DOCS", name: "Doximity Inc." },
    { ticker: "HIMS", name: "Hims & Hers Health" },
  ],
  "E-commerce": [
    { ticker: "SHOP", name: "Shopify Inc." },
    { ticker: "ETSY", name: "Etsy Inc." },
    { ticker: "CHWY", name: "Chewy Inc." },
    { ticker: "SE", name: "Sea Limited" },
    { ticker: "MELI", name: "MercadoLibre Inc." },
  ],
  "Space & Quantum": [
    { ticker: "RKLB", name: "Rocket Lab USA" },
    { ticker: "ASTS", name: "AST SpaceMobile" },
    { ticker: "IRDM", name: "Iridium Communications" },
    { ticker: "IONQ", name: "IonQ Inc." },
  ],
}

function Screener() {
  const [days, setDays] = useState('5')
  const [minConfidence, setMinConfidence] = useState(75)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showStockList, setShowStockList] = useState(false)

  const daysNum = Number(days)
  const isValidDays = days !== '' && daysNum >= 1 && daysNum <= 30

  // Count total stocks
  const totalStocks = Object.values(STOCK_CATEGORIES).flat().length

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!isValidDays) return

    setLoading(true)
    setResult(null)
    setError(null)

    try {
      const response = await fetch(
        `${API_BASE}/screener?days=${daysNum}&min_confidence=${minConfidence / 100}`
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
      <p className="card-description">
        Scans all stocks and ETFs to find those predicted to go <strong>up</strong> with
        confidence above your chosen threshold. Results are sorted by confidence level.
      </p>
      <button
        className="view-stocks-button"
        onClick={() => setShowStockList(true)}
      >
        <span className="view-stocks-icon">📋</span>
        View all {totalStocks} stocks & ETFs scanned →
      </button>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="screener-days">Days Ahead (1-30)</label>
          <input
            id="screener-days"
            type="number"
            min="1"
            max="30"
            value={days}
            onChange={(e) => setDays(e.target.value)}
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

        <button type="submit" disabled={loading || !isValidDays}>
          {loading && <span className="spinner"></span>}
          {loading ? 'Scanning...' : 'Scan Stocks'}
        </button>
      </form>

      {loading && (
        <p className="loading-text">Scanning {totalStocks} stocks & ETFs... This may take ~90 seconds.</p>
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
                    <th>Stock</th>
                    <th>Price</th>
                    <th>Predicted</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {result.matches.map((stock) => (
                    <tr key={stock.ticker}>
                      <td>
                        <strong>{stock.ticker}</strong>
                        <div className="stock-name">{stock.name}</div>
                      </td>
                      <td>${stock.latest_price}</td>
                      <td>
                        ${stock.predicted_price}
                        <span className="change-up"> (+{stock.predicted_change}%)</span>
                      </td>
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

      {/* Stock List Modal */}
      {showStockList && (
        <div className="modal-overlay" onClick={() => setShowStockList(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Stocks & ETFs Scanned ({totalStocks})</h3>
              <button className="modal-close" onClick={() => setShowStockList(false)}>×</button>
            </div>
            <div className="modal-body">
              {Object.entries(STOCK_CATEGORIES).map(([category, stocks]) => (
                <div key={category} className="stock-category">
                  <h4>{category}</h4>
                  <div className="stock-list-items">
                    {stocks.map(stock => (
                      <div key={stock.ticker} className="stock-list-item">
                        <span className="stock-list-ticker">{stock.ticker}</span>
                        <span className="stock-list-name">{stock.name}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Screener
