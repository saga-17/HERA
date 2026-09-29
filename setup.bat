@echo off
echo === HERA-VLM Setup ===

python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo Installing backend dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

if not exist ".env" (
    copy .env.example .env
    echo Created .env from .env.example (demo mode enabled)
)

if not exist "data\uploads" mkdir data\uploads
if not exist "data\results" mkdir data\results

where npm >nul 2>&1
if %errorlevel% equ 0 (
    echo Installing frontend dependencies...
    cd frontend
    call npm install
    cd ..
) else (
    echo WARNING: npm not found. Install Node.js to set up the frontend.
)

echo.
echo === Setup Complete ===
echo.
echo To run in demo mode (no GPU required):
echo   .venv\Scripts\activate
echo   set HERA_DEMO_MODE=true
echo   uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo In another terminal:
echo   cd frontend ^&^& npm run dev
echo.
echo Open http://localhost:5173
