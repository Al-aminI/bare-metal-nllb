#!/bin/bash
# Setup script for MetalNLLB

set -e

echo "==================================="
echo "MetalNLLB Setup"
echo "==================================="
echo

# Check for required tools
echo "Checking dependencies..."
command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is required but not installed."; exit 1; }
command -v make >/dev/null 2>&1 || { echo "Error: make is required but not installed."; exit 1; }
command -v cc >/dev/null 2>&1 || { echo "Error: C compiler (cc) is required but not installed."; exit 1; }

echo "✓ All dependencies found"
echo

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate and install Python packages
echo
echo "Installing Python packages..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q transformers ctranslate2 safetensors numpy

echo "✓ Python packages installed"
echo

# Download and convert model
if [ ! -f "model_int8_ct2.safetensors" ]; then
    echo "Downloading NLLB-200 model..."
    echo "This may take a few minutes (model is ~1.1GB)..."
    
    # Create temporary conversion script
    cat > /tmp/convert_model.py << 'EOF'
import ctranslate2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

print("Loading model from Hugging Face...")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

print("Converting to CTranslate2 format...")
converter = ctranslate2.converters.TransformersConverter(
    "facebook/nllb-200-distilled-600M"
)
converter.convert("/tmp/nllb-200-600M-ct2-int8", quantization="int8")

print("Extracting SafeTensors...")
# The model is now in CT2 format at /tmp/nllb-200-600M-ct2-int8
print("✓ Model ready at /tmp/nllb-200-600M-ct2-int8")
EOF

    python /tmp/convert_model.py
    rm /tmp/convert_model.py
    
    echo "✓ Model downloaded and converted"
else
    echo "✓ Model already exists"
fi

echo

# Build the engine
echo "Building MetalNLLB..."
make clean
make optimized

echo
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo
echo "Quick Start:"
echo "  ./pico_nllb_opt model_int8_ct2.safetensors eng_Latn hau_Latn 59002 4 2"
echo
echo "Run tests:"
echo "  source venv/bin/activate"
echo "  python benchmarks/test_hausa_translation.py"
echo
echo "See README.md for more information."
