#!/bin/bash

# Accident Detection System Setup Script
# This script sets up all required configurations and dependencies

set -e  # Exit on error

echo "==================================================="
echo "Accident Detection System - Setup Script"
echo "==================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.6 or higher."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo "✓ pip upgraded"
echo ""

# Install requirements
echo "Installing required packages from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ All packages installed successfully"
else
    echo "❌ requirements.txt not found in current directory"
    exit 1
fi

echo ""

# Create data directory structure if not exists
echo "Setting up data directories..."
mkdir -p data/train/Accident
mkdir -p data/train/Non\ Accident
mkdir -p data/val/Accident
mkdir -p data/val/Non\ Accident
mkdir -p data/test/Accident
mkdir -p data/test/Non\ Accident
echo "✓ Data directories created"
echo ""

# Create output directories
echo "Creating output directories..."
mkdir -p outputs
mkdir -p logs
echo "✓ Output directories created"
echo ""

# Check for model files
echo "Checking for model files..."
if [ ! -f "model.json" ] || [ ! -f "model_weights.h5" ]; then
    echo "⚠️  Model files (model.json, model_weights.h5) not found."
    echo "   Please run the accident-classification.ipynb notebook to generate these files."
else
    echo "✓ Model files found"
fi

echo ""
echo "==================================================="
echo "✓ Setup completed successfully!"
echo "==================================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. If model files are missing, run: jupyter notebook accident-classification.ipynb"
echo "3. To run the application: python main.py"
echo "4. To run with Streamlit: streamlit run streamlit_app.py"
echo ""
