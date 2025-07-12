#!/bin/bash
# Setup script for Better Memory MCP with semantic search

set -e

echo "Better Memory MCP Setup Script"
echo "=============================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    echo "Please install Python 3.8 or later and try again."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python $REQUIRED_VERSION or later is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

# Check if pip is installed
if ! python3 -m pip --version &> /dev/null; then
    echo "Error: pip is not installed."
    echo "Please install pip and try again."
    exit 1
fi

echo "✓ pip is installed"

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo ""
echo "Installing Python dependencies for semantic search..."
echo "This may take a few minutes on first install..."

# Check if CUDA is available and install appropriate faiss
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA detected, installing faiss-gpu..."
    pip install faiss-gpu
else
    echo "No CUDA detected, installing faiss-cpu..."
    pip install faiss-cpu
fi

# Install other requirements
pip install -r requirements.txt

# Download model weights in advance
echo ""
echo "Pre-downloading ModernColBERT model weights..."
echo "This ensures faster first-time startup..."
python3 -c "
from transformers import AutoTokenizer, AutoModel
print('Downloading model: lightonai/Reason-ModernColBERT')
tokenizer = AutoTokenizer.from_pretrained('lightonai/Reason-ModernColBERT', cache_dir='./cache')
model = AutoModel.from_pretrained('lightonai/Reason-ModernColBERT', cache_dir='./cache', trust_remote_code=True)
print('✓ Model downloaded successfully')
"

# Install Node.js dependencies
echo ""
echo "Installing Node.js dependencies..."
npm install

# Build TypeScript
echo ""
echo "Building TypeScript..."
npm run build

echo ""
echo "✓ Setup complete!"
echo ""
echo "To use semantic search, make sure to run the MCP server with Python available."
echo "The semantic search service will start automatically when the MCP server starts."
echo ""
echo "If semantic search fails to start, the MCP server will continue to work"
echo "with regular keyword search as a fallback."