# Quick Start Guide

Get MetalNLLB running in 5 minutes!

## 1. Clone the Repository

```bash
git clone https://github.com/Al-aminI/bare-metal-nllb
cd MetalNLLB
```

## 2. Run Setup

```bash
./setup.sh
```

This will:
- Check dependencies (python3, make, cc)
- Create Python virtual environment
- Install required packages (transformers, ctranslate2)
- Download NLLB-200 model (~1.1GB)
- Build the optimized engine

## 3. Test Translation

```bash
# Translate "Hello." from English to Hausa
./pico_nllb_opt model_int8_ct2.safetensors eng_Latn hau_Latn 59002 4 2

# Output: Barka dai.
```

## 4. Run Quality Tests

```bash
source venv/bin/activate
python benchmarks/test_hausa_translation.py

# Expected: 5/5 exact matches (100% parity)
```

## 5. Run Performance Benchmarks

```bash
# Compare with CTranslate2
python benchmarks/benchmark_direct.py

# Expected: ~1.46x faster than CT2
```

## Common Commands

### Translate Text

```bash
# English to French
./pico_nllb_opt model_int8_ct2.safetensors eng_Latn fra_Latn 59002 4 2

# English to Spanish
./pico_nllb_opt model_int8_ct2.safetensors eng_Latn spa_Latn 59002 4 2

# English to Arabic
./pico_nllb_opt model_int8_ct2.safetensors eng_Latn ara_Arab 59002 4 2
```

### Build Variants

```bash
# Optimized (default)
make optimized

# Baseline (no optimizations)
make baseline

# Clean build
make clean && make optimized
```

## Supported Languages

MetalNLLB supports 200+ languages. Common ones:

| Language | Code | Example |
|----------|------|---------|
| English | `eng_Latn` | Hello |
| French | `fra_Latn` | Bonjour |
| Spanish | `spa_Latn` | Hola |
| German | `deu_Latn` | Hallo |
| Chinese | `zho_Hans` | 你好 |
| Arabic | `ara_Arab` | مرحبا |
| Hausa | `hau_Latn` | Barka dai |
| Yoruba | `yor_Latn` | Ẹ káàbọ̀ |
| Swahili | `swh_Latn` | Habari |

See [NLLB-200 documentation](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200) for full list.

## Tokenization

To translate arbitrary text, you need to tokenize it first:

```python
from transformers import NllbTokenizer

tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer.src_lang = "eng_Latn"

# Tokenize
text = "The scientific method is a systematic way of learning."
tokens = tokenizer.encode(text, add_special_tokens=True)

# Remove source language token (engine adds it)
src_token = tokenizer.convert_tokens_to_ids("eng_Latn")
input_tokens = [t for t in tokens if t != src_token]

# Run engine
# ./pico_nllb_opt model_int8_ct2.safetensors eng_Latn hau_Latn [tokens...]
```

## Troubleshooting

### Model Not Found

```bash
# Download model manually
python -c "
import ctranslate2
converter = ctranslate2.converters.TransformersConverter('facebook/nllb-200-distilled-600M')
converter.convert('/tmp/nllb-200-600M-ct2-int8', quantization='int8')
"
```

### Build Errors

```bash
# macOS: Install Xcode Command Line Tools
xcode-select --install

# Linux: Install build essentials
sudo apt-get install build-essential
```

### Python Errors

```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install transformers ctranslate2 safetensors numpy
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Explore [research_report/](research_report/) for technical details
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- See [docs/FINAL_BENCHMARK.md](docs/FINAL_BENCHMARK.md) for performance analysis

## Performance Tips

1. **Use Optimized Build**: `make optimized` (default)
2. **Multi-Core**: Engine automatically uses 4 cores
3. **ARM/NEON**: On Raspberry Pi, NEON SIMD provides 4-8x speedup
4. **Batch Processing**: Process multiple sentences in parallel (future work)

## Getting Help

- Open an issue on GitHub
- Check existing issues for solutions
- Read the documentation in `docs/` and `research_report/`

Happy translating! 🚀
