# 2. Background and Related Work

## 2.1 Neural Machine Translation

### 2.1.1 Transformer Architecture
The transformer architecture (Vaswani et al., 2017) revolutionized NMT through:
- Self-attention mechanisms for long-range dependencies
- Parallel processing of sequences
- Encoder-decoder structure for sequence-to-sequence tasks

### 2.1.2 NLLB-200 Model
Meta's NLLB-200 (Costa-jussà et al., 2022) represents state-of-the-art multilingual NMT:
- **Parameters:** 615M (distilled from 1.3B)
- **Languages:** 200 languages with 40,200 translation directions
- **Architecture:** 12-layer encoder, 12-layer decoder
- **Vocabulary:** 256,206 tokens (shared across all languages)
- **Embedding dimension:** 1024
- **FFN dimension:** 4096

## 2.2 Model Quantization

### 2.2.1 Quantization Schemes

**Full Precision (FP32):**
- Standard training format
- 4 bytes per parameter
- 2.4GB for 600M parameters

**Half Precision (FP16):**
- 2 bytes per parameter
- 1.2GB model size
- Minimal quality loss

**INT8 Quantization:**
- 1 byte per parameter + scales
- ~600MB model size
- Requires calibration for quality

**NF4 (4-bit Normal Float):**
- 0.5 bytes per parameter + scales
- ~300MB model size
- Information-theoretically optimal for normal distributions

### 2.2.2 Quantization-Aware Training vs Post-Training Quantization

**QAT (Quantization-Aware Training):**
- Simulates quantization during training
- Better quality preservation
- Requires retraining

**PTQ (Post-Training Quantization):**
- Applied to pre-trained models
- No retraining required
- Used in this study

## 2.3 Existing Inference Frameworks

### 2.3.1 CTranslate2
- Optimized C++ inference engine
- INT8/INT16 quantization support
- 4x faster than PyTorch
- Industry standard for production NMT

### 2.3.2 ONNX Runtime
- Cross-platform inference
- Multiple quantization backends
- Heavier dependencies

### 2.3.3 llama.cpp / ggml
- Bare-metal LLM inference
- Inspired our approach
- Decoder-only (not encoder-decoder)

## 2.4 Gap in Literature

**Missing:** Comprehensive documentation of bare-metal encoder-decoder transformer implementation with:
- Complete bug taxonomy
- Quantization scheme comparison
- Production-ready code
- Exact parity validation

This study fills that gap.
