#!/bin/bash
cd "$(dirname "$0")"

# Start backend in background
source venv/bin/activate
uvicorn app:app --port 8000 &
BACKEND_PID=$!

# Start frontend
cd frontend
npm run dev

# When frontend exits, kill backend
kill $BACKEND_PID
