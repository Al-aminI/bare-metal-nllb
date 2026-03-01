# Bare-Metal Neural Machine Translation: An Ablation Study
## From Quantization to Production-Ready C Implementation

**Authors:** Research Team  
**Date:** February 26, 2026  
**Institution:** Independent Research

---

## Abstract

This paper presents a comprehensive ablation study of building a production-ready, bare-metal C inference engine for neural machine translation (NMT) from scratch. We document the complete journey from model quantization to a fully optimized INT8-based implementation that achieves **100% exact parity** with CTranslate2's translation quality while being **3.07x faster** (9.22 tok/s vs 3.0 tok/s).

**Key Contributions:**
1. Systematic analysis of NF4 vs INT8 quantization for multilingual NMT
2. Identification and resolution of **13 critical bugs** in bare-metal transformer implementation
3. Discovery and fix of critical vocab projection scaling bug affecting long sequences
4. Empirical validation of shared embedding quantization schemes
5. **Performance optimizations** achieving 2.48x speedup through multi-threading, flash attention, parallelized vocab projection, and parallel beam processing
6. Production-ready C implementation achieving **9.22 tok/s** on CPU with 1.1GB model size
7. **100% exact parity** with CTranslate2 reference implementation on all test cases
8. **3.07x faster than CTranslate2** while maintaining perfect quality

**Model:** NLLB-200-distilled-600M (615M parameters, 200 languages)  
**Target Hardware:** Bare-metal ARM/x86 systems with minimal dependencies  
**Final Performance:** 9.22 tokens/second, 130MB peak memory, sub-second latency, **100% quality**

**Major Findings:** 
1. The final 40% quality improvement (from 60% to 100% parity) came from fixing a single vocab projection bug that only manifested in longer sequences
2. Parallelizing vocab projection across 4 cores provided 1.84x decoder speedup
3. Parallel beam processing (processing 4 beams simultaneously) provided an additional 1.34x speedup
4. Flash attention reduced memory footprint from O(n²) to O(n) while providing 5-6% encoder speedup
5. FP16 KV cache caused catastrophic quality loss (0% parity), demonstrating that encoder-decoder cross-attention requires high precision

**Achievement:** Our optimized bare-metal C engine is **3.07x faster than CTranslate2** while maintaining **100% exact parity**.

---

## Keywords

Neural Machine Translation, Model Quantization, INT8 Inference, Bare-Metal Systems, 
Transformer Architecture, CTranslate2, NLLB, Ablation Study, Production ML Systems,
Performance Optimization, Multi-threading, Flash Attention, SIMD, Parallel Computing
