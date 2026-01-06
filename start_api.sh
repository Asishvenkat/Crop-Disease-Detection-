#!/bin/bash
# ============================================================================
# Crop Disease Detection - Startup Script for Linux/Mac
# ============================================================================

set -e

echo ""
echo "============================================================================"
echo "🌱 CROP DISEASE DETECTION - STARTUP SCRIPT"
echo "============================================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "   Please install Python 3.9+ "
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Check if model exists
if [ ! -f "models/pdd.h5" ]; then
    echo "⚠️  Model not found!"
    echo "   You need to train the model first."
    echo ""
    echo "Run: python train.py"
    echo "(This requires the dataset in data/dataset/)"
    echo ""
    exit 1
fi

echo "✓ Model found"
echo ""

# Check dependencies
echo "📚 Checking dependencies..."
pip install -q -r requirements.txt 2>/dev/null || pip install -r requirements.txt
echo "✓ Dependencies ready"
echo ""

# Start API
echo "============================================================================"
echo "🚀 Starting API Server..."
echo "============================================================================"
echo ""
echo "📝 API will run on: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "In a new terminal, run: streamlit run app.py"
echo ""
echo "============================================================================"
echo ""

python main.py
