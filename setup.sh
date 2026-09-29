#!/usr/bin/env bash
set -euo pipefail

echo "=== HERA-VLM Setup ==="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing backend dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy env file if missing
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env from .env.example (demo mode enabled)"
fi

# Create data directories
mkdir -p data/uploads data/results

# Frontend setup
if command -v npm &> /dev/null; then
    echo "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
else
    echo "WARNING: npm not found. Install Node.js to set up the frontend."
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run in demo mode (no GPU required):"
echo "  source .venv/bin/activate"
echo "  HERA_DEMO_MODE=true uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "In another terminal:"
echo "  cd frontend && npm run dev"
echo ""
echo "Open http://localhost:5173"
