#!/bin/bash

# FIDES Setup Script for Swathi
# Installs all dependencies and validates environment

echo "════════════════════════════════════════════════════════════════════════════"
echo "FIDES ENVIRONMENT SETUP"
echo "════════════════════════════════════════════════════════════════════════════"

# Step 1: Check Python version
echo ""
echo "Step 1: Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 9) else 1)'; then
    echo "❌ Python 3.9+ required"
    exit 1
fi
echo "✓ Python 3.9+ detected"

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Step 3: Activate virtual environment
echo ""
echo "Step 3: Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Step 4: Upgrade pip
echo ""
echo "Step 4: Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ pip upgraded"

# Step 5: Install dependencies
echo ""
echo "Step 5: Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Step 6: Validate installation
echo ""
echo "Step 6: Validating installation..."
python3 validate_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "════════════════════════════════════════════════════════════════════════════"
    echo "✅ SETUP COMPLETE"
    echo "════════════════════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo "1. Activate environment: source venv/bin/activate"
    echo "2. Load your MIMIC-IV data into: results/disease_cohorts/"
    echo "3. Read HANDOFF.md for step-by-step instructions"
    echo "4. Run: python experiments/run_fides_real_causal_discovery.py"
    echo ""
else
    echo ""
    echo "❌ Validation failed. See errors above."
    exit 1
fi
