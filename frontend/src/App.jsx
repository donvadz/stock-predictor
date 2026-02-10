import { useState } from 'react'
import RegimeBanner from './components/RegimeBanner.jsx'
import OptimalScan from './components/OptimalScan.jsx'
import PredictForm from './components/PredictForm.jsx'
import Screener from './components/Screener.jsx'
import Backtest from './components/Backtest.jsx'
import ScreenerBacktest from './components/ScreenerBacktest.jsx'
import RegimeGatedBacktest from './components/RegimeGatedBacktest.jsx'
import StressTest from './components/StressTest.jsx'

function App() {
  const [showAbout, setShowAbout] = useState(false)
  const [activeTab, setActiveTab] = useState('signals')

  return (
    <div className="app">
      {/* Regime Banner - Always visible */}
      <RegimeBanner />

      <header>
        <h1>Stock Predictor</h1>
        <p>Multi-signal consensus trading strategy</p>
        <p className="header-caveat">
          78% accuracy on Tier 1 stocks (2023-2024 bull market) | Fails during crises
        </p>
      </header>

      {/* Tab Navigation */}
      <nav className="tab-nav">
        <button
          className={`tab-button ${activeTab === 'signals' ? 'active' : ''}`}
          onClick={() => setActiveTab('signals')}
        >
          <span className="tab-icon">🎯</span>
          Signals
        </button>
        <button
          className={`tab-button ${activeTab === 'screener' ? 'active' : ''}`}
          onClick={() => setActiveTab('screener')}
        >
          <span className="tab-icon">📊</span>
          Screener
        </button>
        <button
          className={`tab-button ${activeTab === 'backtest' ? 'active' : ''}`}
          onClick={() => setActiveTab('backtest')}
        >
          <span className="tab-icon">📈</span>
          Backtest
        </button>
        <button
          className={`tab-button ${activeTab === 'stress' ? 'active' : ''}`}
          onClick={() => setActiveTab('stress')}
        >
          <span className="tab-icon">⚠️</span>
          Stress Test
        </button>
      </nav>

      <main>
        {/* Signals Tab - Primary */}
        {activeTab === 'signals' && (
          <div className="tab-content signals-tab">
            <OptimalScan />
            <PredictForm />
          </div>
        )}

        {/* Screener Tab */}
        {activeTab === 'screener' && (
          <div className="tab-content screener-tab">
            <Screener />
          </div>
        )}

        {/* Backtest Tab */}
        {activeTab === 'backtest' && (
          <div className="tab-content backtest-tab">
            <Backtest />
            <ScreenerBacktest />
            <RegimeGatedBacktest />
          </div>
        )}

        {/* Stress Test Tab */}
        {activeTab === 'stress' && (
          <div className="tab-content stress-tab">
            <StressTest />
          </div>
        )}
      </main>

      <footer>
        <button className="about-button" onClick={() => setShowAbout(true)}>
          Limitations & How It Works
        </button>
      </footer>

      {/* About Modal - Updated with honest limitations */}
      {showAbout && (
        <div className="modal-overlay" onClick={() => setShowAbout(false)}>
          <div className="modal about-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Honest Limitations</h3>
              <button className="modal-close" onClick={() => setShowAbout(false)}>×</button>
            </div>
            <div className="modal-body">
              <section className="about-section warning-section">
                <h4>⚠️ What This System CANNOT Do</h4>
                <ul>
                  <li>Predict with 89% accuracy across all conditions</li>
                  <li>Work reliably during market crashes</li>
                  <li>Generate daily trading signals</li>
                  <li>Replace human judgment on position sizing and risk</li>
                  <li>Guarantee any returns</li>
                </ul>
              </section>

              <section className="about-section">
                <h4>📊 Accuracy Claims - Context</h4>
                <table className="about-table">
                  <thead>
                    <tr>
                      <th>Period</th>
                      <th>Accuracy</th>
                      <th>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="row-good">
                      <td>2023-2024 (bull)</td>
                      <td>78%</td>
                      <td>Tier 1 stocks, 4+ signals</td>
                    </tr>
                    <tr className="row-ok">
                      <td>2022 (bear)</td>
                      <td>57%</td>
                      <td>Survived but degraded</td>
                    </tr>
                    <tr className="row-bad">
                      <td>2020 COVID</td>
                      <td>29%</td>
                      <td>FAILED</td>
                    </tr>
                    <tr className="row-bad">
                      <td>2018 Q4</td>
                      <td>33%</td>
                      <td>FAILED</td>
                    </tr>
                    <tr className="row-bad">
                      <td>2008 Crisis</td>
                      <td>31%</td>
                      <td>FAILED</td>
                    </tr>
                  </tbody>
                </table>
              </section>

              <section className="about-section">
                <h4>🎯 What This System DOES</h4>
                <ul>
                  <li>Identifies stocks with aligned technical signals</li>
                  <li>Uses multi-signal consensus (ML + trend + volatility + momentum + RSI)</li>
                  <li>Works best with 20-day holding period</li>
                  <li>Generates 1-2 high-conviction signals per month</li>
                  <li>Tells you when NOT to trade (regime detection)</li>
                </ul>
              </section>

              <section className="about-section">
                <h4>✅ Before Using Real Money</h4>
                <ul>
                  <li><strong>Check regime first</strong> - Only trade when NORMAL</li>
                  <li><strong>Paper trade 3-6 months</strong> - Verify it works for you</li>
                  <li><strong>Position sizing</strong> - Never risk more than you can afford to lose</li>
                  <li><strong>Diversification</strong> - This is ONE input, not your only signal</li>
                </ul>
              </section>

              <section className="about-section">
                <h4>🤖 Technical Details</h4>
                <p>
                  Uses a <strong>Random Forest</strong> ensemble (200 trees) trained on 23 technical
                  indicators and 27 fundamental features. Backtested using realistic methodology
                  (train once, freeze, test separately - no data leakage).
                </p>
              </section>

              <section className="about-section disclaimer-section">
                <h4>⚖️ Disclaimer</h4>
                <p>
                  This tool is for <strong>educational and research purposes only</strong>.
                  Past performance does not guarantee future results. Stock markets are
                  inherently unpredictable. Never invest based solely on algorithmic predictions.
                  Always do your own research and consider consulting a financial advisor.
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
