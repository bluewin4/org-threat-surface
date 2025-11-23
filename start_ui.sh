#!/bin/bash

# Organization Threat Surface Simulator - Quick Start Script

set -e

echo "🏢 Organization Threat Surface Simulator"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "Simulations/main.py" ]; then
    echo "❌ Error: main.py not found in Simulations/"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "✓ Found simulation code"

# Check for data files
if [ ! -f "Simulations/master_SP500_TMT.csv" ]; then
    echo "⚠️  Warning: master_SP500_TMT.csv not found"
    echo "   Some features will be disabled"
fi

if [ ! -f "Simulations/snapshot.csv" ]; then
    echo "⚠️  Warning: snapshot.csv not found"
    echo "   Some features will be disabled"
fi

# Install or upgrade dependencies
echo ""
echo "📦 Installing UI dependencies..."
pip install -q -r Simulations/requirements_ui.txt
echo "✓ Dependencies installed"

# Launch Streamlit
echo ""
echo "🚀 Starting web server..."
echo "   Opening http://localhost:8501 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd Simulations
streamlit run app.py
