import PredictForm from './components/PredictForm.jsx'
import Screener from './components/Screener.jsx'

function App() {
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
    </div>
  )
}

export default App
