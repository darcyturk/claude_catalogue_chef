#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧑‍🍳 Welcome to Catalog Chef Setup! 🧑‍🍳${NC}"
echo "Setting up your PDF extraction kitchen..."

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is required but not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 3 found${NC}"

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 is required but not installed.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ pip3 found${NC}"

# Create virtual environment
echo -e "${YELLOW}📦 Creating virtual environment...${NC}"
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo -e "${YELLOW}📦 Installing Python dependencies...${NC}"
pip install --upgrade pip

# Core dependencies
pip install PyMuPDF pdfplumber Pillow
pip install playwright transformers torch datasets
pip install opencv-python pyzbar fuzzywuzzy
pip install jsondiff PyQt6

# Install Playwright browsers
echo -e "${YELLOW}🌐 Installing Playwright browsers...${NC}"
playwright install

# Create directory structure
echo -e "${YELLOW}📁 Creating project directories...${NC}"
mkdir -p data/raw_pdfs
mkdir -p data/processed_pages
mkdir -p data/images/products
mkdir -p data/images/brands
mkdir -p data/learning_data
mkdir -p data/ground_truth
mkdir -p models/layoutlm_finetuned
mkdir -p models/barcode_detector_weights
mkdir -p output
mkdir -p temp
mkdir -p logs

# Create initial state file
echo '{"session_id": null, "current_step": "start", "total_pages": 0, "processed_pages": 0, "corrections_made": 0}' > state.json

# Create initial config
cat > config/settings.py << 'EOF'
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
TEMP_DIR = os.path.join(PROJECT_ROOT, 'temp')

# LayoutLM Settings
LAYOUTLM_MODEL = "microsoft/layoutlmv3-base"
LAYOUTLM_CONFIDENCE_THRESHOLD = 0.85

# OCR Settings
TESSERACT_PATH = "/usr/bin/tesseract"  # Adjust based on system
OCR_CONFIDENCE_THRESHOLD = 60

# Web Scraping Settings
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
SEARCH_DELAY = 2  # seconds between searches

# Image Processing
MIN_LOGO_SIZE = (50, 50)
MAX_LOGO_SIZE = (500, 500)
LOGO_ASPECT_RATIO_RANGE = (0.3, 3.0)

# JSON Schema
PRODUCT_SCHEMA = {
    "page_number": int,
    "sku": str,
    "product_name": str,
    "images": list,
    "prices": list,
    "variations": list,
    "minimum_order_quantity": str,
    "barcodes": list,
    "related_products": list
}
EOF

echo -e "${GREEN}✅ Setup complete! Your Catalog Chef kitchen is ready.${NC}"
echo -e "${BLUE}🚀 Run './run.sh' to start cooking!${NC}"

# Instructions for Tesseract
echo -e "${YELLOW}📝 NOTE: Install Tesseract-OCR for enhanced text extraction:${NC}"
echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr"
echo "  macOS: brew install tesseract"
echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
