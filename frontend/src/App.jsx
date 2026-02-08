import { useState } from 'react'
import PredictForm from './components/PredictForm.jsx'
import Screener from './components/Screener.jsx'

function App() {
  const [showAbout, setShowAbout] = useState(false)

  return (
    <div className="app">
      <header>
        <h1>Stock Predictor</h1>
        <p>ML-powered stock direction predictions</p>
      </header>

      <main>
        <PredictForm />
        <Screener />
      </main>

      <footer>
        <button className="about-button" onClick={() => setShowAbout(true)}>
          How does this work?
        </button>
      </footer>

      {/* About Modal */}
      {showAbout && (
        <div className="modal-overlay" onClick={() => setShowAbout(false)}>
          <div className="modal about-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>How It Works</h3>
              <button className="modal-close" onClick={() => setShowAbout(false)}>×</button>
            </div>
            <div className="modal-body">
              <section className="about-section">
                <h4>🤖 Machine Learning Model</h4>
                <p>
                  This app uses a <strong>Random Forest</strong> ensemble model with 200 decision trees
                  to predict stock price direction. A separate regression model predicts the expected
                  percentage change.
                </p>
              </section>

              <section className="about-section">
                <h4>📊 Technical Indicators (23 features)</h4>
                <ul>
                  <li><strong>Price Returns:</strong> 1-day, 5-day, and 20-day returns</li>
                  <li><strong>Volatility:</strong> 5-day and 20-day rolling standard deviation</li>
                  <li><strong>Moving Averages:</strong> SMA ratios (5/20, 20/50, 50/200), price-to-SMA</li>
                  <li><strong>Volume:</strong> Volume change, volume ratio to 20-day average</li>
                  <li><strong>RSI:</strong> 14-day Relative Strength Index</li>
                  <li><strong>MACD:</strong> Moving Average Convergence Divergence + signal line</li>
                  <li><strong>Bollinger Bands:</strong> Upper/lower band distance, bandwidth</li>
                  <li><strong>ATR:</strong> 14-day Average True Range (volatility)</li>
                  <li><strong>OBV:</strong> On-Balance Volume trend</li>
                </ul>
              </section>

              <section className="about-section">
                <h4>📈 Fundamental Data (27 features)</h4>
                <ul>
                  <li><strong>Valuation:</strong> P/E ratio, Forward P/E, PEG, Price/Book, Price/Sales</li>
                  <li><strong>Profitability:</strong> Profit margin, Operating margin, ROE, ROA</li>
                  <li><strong>Growth:</strong> Revenue growth, Earnings growth</li>
                  <li><strong>Financial Health:</strong> Debt/Equity, Current ratio, Quick ratio</li>
                  <li><strong>Analyst Data:</strong> Sentiment score, recommendation count</li>
                  <li><strong>Market Data:</strong> Beta, 52-week change, Short interest</li>
                  <li><strong>Other:</strong> Sector, Market cap, Earnings surprise, News count</li>
                </ul>
                <p className="about-note">
                  Note: ETFs don't have fundamental data, so predictions use technical indicators only.
                </p>
              </section>

              <section className="about-section">
                <h4>📅 Historical Data</h4>
                <ul>
                  <li><strong>Short-term predictions (1-7 days):</strong> Uses 1 year of daily data</li>
                  <li><strong>Long-term predictions (8-30 days):</strong> Uses 3 years of daily data</li>
                </ul>
              </section>

              <section className="about-section">
                <h4>⚠️ Disclaimer</h4>
                <p>
                  This tool is for <strong>educational and research purposes only</strong>. Stock markets are
                  inherently unpredictable, and even sophisticated models rarely achieve more than 55-60%
                  accuracy consistently. Never invest based solely on algorithmic predictions. Always do
                  your own research and consider consulting a financial advisor.
                </p>
              </section>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
