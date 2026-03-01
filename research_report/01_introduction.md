# 1. Introduction

## 1.1 Motivation

Modern neural machine translation systems achieve remarkable quality but require substantial computational resources. State-of-the-art models like NLLB-200 (No Language Left Behind) support 200 languages but typically require:
- 2.4GB+ of RAM for full precision inference
- GPU acceleration for acceptable latency
- Complex runtime dependencies (PyTorch, CUDA, etc.)

This creates a deployment gap for:
- Edge devices (Raspberry Pi, embedded systems)
- Resource-constrained environments
- Offline/air-gapped systems
- Cost-sensitive applications

## 1.2 Research Questions

This study addresses five fundamental questions:

**RQ1:** Can aggressive quantization (4-bit NF4 or 8-bit INT8) maintain translation quality for multilingual models?

**RQ2:** What are the critical implementation challenges in bare-metal transformer inference?

**RQ3:** How do different quantization schemes affect cross-attention quality in encoder-decoder architectures?

**RQ4:** Can a from-scratch C implementation achieve parity with optimized frameworks like CTranslate2?

**RQ5:** What is the minimal memory footprint achievable while maintaining production-quality translations?

## 1.3 Approach

We adopt a systematic ablation methodology:
1. **Baseline establishment:** CTranslate2 INT8 as ground truth
2. **Iterative implementation:** Build each component with validation
3. **Bug isolation:** Identify and document each failure mode
4. **Quantitative comparison:** Token-level accuracy against reference
5. **Performance profiling:** Latency and memory measurements

## 1.4 Contributions

1. **First complete documentation** of bare-metal NLLB implementation
2. **Identification of 12 critical bugs** in transformer quantization
3. **Novel finding:** Shared embedding quantization requires unified scales
4. **Production-ready codebase:** ~2,500 lines of portable C11
5. **Reproducible methodology:** All experiments documented with exact commands
