#!/bin/bash
# HRCS Installation Script
# For Linux/Mac systems

set -e

echo "HRCS - Harmonic Resonance Communication System"
echo "Installation Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install package in development mode
echo "Installing HRCS package..."
pip install -e .

# Create config if doesn't exist
echo ""
if [ ! -f "config/config.yaml" ]; then
    cp config/config.yaml.example config/config.yaml
    echo "✅ Created config/config.yaml from template"
    echo "⚠️  Remember to edit config/config.yaml before use!"
else
    echo "ℹ️  config/config.yaml already exists"
fi

# Create log directory
mkdir -p logs

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Configure node: python scripts/setup_node.py"
echo "3. Start node: python -m hrcs.node"
echo "4. Or use CLI: hrcs status"
echo ""
echo "For more information, see README.md"

