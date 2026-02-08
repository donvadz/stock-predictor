# Stock Direction Predictor

A minimal FastAPI service that predicts stock price direction (up/down) using machine learning.

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run Full Stack (Recommended)

```bash
cd frontend
npm install
npm run dev:full
```

This starts both the backend (port 8000) and frontend (port 5173) together.

### Run Backend Only

```bash
uvicorn app:app --reload --port 8000
```

Make a prediction:
```bash
curl "http://localhost:8000/predict?ticker=AAPL&days=5"
```

### Response Format

```json
{
  "ticker": "AAPL",
  "days": 5,
  "direction": "up",
  "confidence": 0.65
}
```

### Parameters

- `ticker` (required): Stock ticker symbol (e.g., AAPL, GOOGL, MSFT)
- `days` (required): Prediction horizon in days (1-30)

### Error Codes

- `400`: Insufficient historical data for training
- `404`: Invalid ticker or no data available
- `422`: Invalid parameters (e.g., days out of range)
- `500`: Prediction failed

## How It Works

1. Fetches 1 year of daily OHLCV data from Yahoo Finance
2. Computes technical features (returns, volatility, moving averages, volume metrics)
3. Trains a RandomForest classifier on historical data
4. Predicts whether the stock will go up or down after N days
5. Returns the prediction with a confidence score

Results are cached for 3 minutes to improve response times.

## Deployment

### Backend (Railway)

1. Push to GitHub
2. Go to [railway.app](https://railway.app) and create a new project
3. Select "Deploy from GitHub repo"
4. Add environment variable: `FRONTEND_URL=https://your-app.vercel.app`
5. Railway auto-detects Python and deploys

### Frontend (Vercel)

1. Go to [vercel.com](https://vercel.com) and import your GitHub repo
2. Set root directory to `frontend`
3. Add environment variable: `VITE_API_URL=https://your-backend.railway.app`
4. Deploy
