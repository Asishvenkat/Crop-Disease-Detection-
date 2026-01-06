@echo off
REM ============================================================================
REM Crop Disease Detection - Startup Script for Windows
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo 🌱 CROP DISEASE DETECTION - STARTUP SCRIPT
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo    Please install Python 3.9+ and add it to PATH
    pause
    exit /b 1
)

echo ✓ Python found
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
    echo.
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)
echo ✓ Virtual environment activated
echo.

REM Check if model exists
if not exist "models\pdd.h5" (
    echo ⚠️  Model not found!
    echo    You need to train the model first.
    echo.
    echo Run: python train.py
    echo (This requires the dataset in data/dataset/)
    echo.
    pause
    exit /b 1
)

echo ✓ Model found
echo.

REM Check dependencies
echo 📚 Checking dependencies...
pip list | find "tensorflow" >nul
if errorlevel 1 (
    echo ⚠️  Dependencies not installed. Installing...
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)
echo ✓ Dependencies ready
echo.

REM Start API
echo ============================================================================
echo 🚀 Starting API Server...
echo ============================================================================
echo.
echo 📝 API will run on: http://localhost:8000
echo 📚 API Docs: http://localhost:8000/docs
echo.
echo In a new terminal, run: streamlit run app.py
echo.
echo ============================================================================
echo.

python main.py

pause
