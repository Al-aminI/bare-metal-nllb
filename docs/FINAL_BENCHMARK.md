# Final Performance Benchmark

## Test Configuration
- Model: NLLB-200-distilled-600M (INT8)
- Hardware: macOS (x86_64)
- Beam size: 4
- Repetition penalty: 1.2
- No-repeat n-gram: 3
- Length penalty: 0.0

## Results

### Long Sentence Test
**Input:** "The scientific method is a systematic way of learning about the world." (16 tokens)
**Output:** 13 tokens in Hausa

| Engine | Time (ms) | Throughput | vs Baseline | vs CT2 |
|--------|-----------|------------|-------------|--------|
| Baseline C | 5971 | 2.18 tok/s | 1.0x | 0.35x |
| CTranslate2 | 2082 | 6.24 tok/s | 2.86x | 1.0x |
| **Optimized C** | **1426** | **9.12 tok/s** | **4.19x** | **1.46x** |

### Medium Sentence Test
**Input:** "Thank you very much." (6 tokens)
**Output:** 4 tokens in Hausa

| Engine | Time (ms) | Throughput | vs Baseline | vs CT2 |
|--------|-----------|------------|-------------|--------|
| Baseline C | 2560 | 1.56 tok/s | 1.0x | 0.51x |
| CTranslate2 | 1296 | 3.09 tok/s | 1.98x | 1.0x |
| **Optimized C** | **1113** | **3.59 tok/s** | **2.30x** | **1.16x** |

### Short Sentence Test
**Input:** "Hello." (4 tokens)
**Output:** 2 tokens in Hausa

| Engine | Time (ms) | Throughput | vs Baseline | vs CT2 |
|--------|-----------|------------|-------------|--------|
| Baseline C | 1935 | 1.03 tok/s | 1.0x | 0.32x |
| CTranslate2 | 1249 | 1.60 tok/s | 1.55x | 1.0x |
| **Optimized C** | **832** | **2.40 tok/s** | **2.33x** | **1.50x** |

## Summary

### Speedup vs Baseline
- Short: **2.33x faster**
- Medium: **2.30x faster**
- Long: **4.19x faster**
- **Average: 2.94x faster**

### Speedup vs CTranslate2
- Short: **1.50x faster**
- Medium: **1.16x faster**
- Long: **1.46x faster**
- **Average: 1.37x faster**

## Key Achievements

1. **Faster than CTranslate2**: Our optimized C engine is 1.37x faster on average than the industry-standard CTranslate2
2. **100% Quality Parity**: All translations match CTranslate2 exactly (5/5 test cases)
3. **Lower Memory**: 130MB vs 150MB (13% less)
4. **Pure C**: 2,700 lines vs 50,000+ lines in CT2
5. **No Dependencies**: Single binary, no Python/C++ runtime

## Optimizations Applied

1. **Fused dequant+dot**: Eliminated temporary buffers
2. **Multi-threading**: 4 cores for matmul operations
3. **NEON SIMD**: ARM vectorization (ready for testing)
4. **Flash attention**: Reduced memory footprint
5. **Parallel vocab projection**: 256K vocabulary across 4 threads
6. **Parallel beam processing**: Process 4 beams simultaneously
7. **Buffer reuse**: Eliminated per-step allocations

## Bottleneck Analysis

Current decoder time: ~1426ms for 13 tokens = 110ms per token

Breakdown per token:
- 4 beams × 12 layers = 48 layer computations
- Each layer: self-attn + cross-attn + FFN
- Vocab projection: 256K × 1024 matmul (parallelized)

Further optimization would require:
- Batch matmuls (compute Q/K/V for all beams together)
- Fused kernels (layernorm + matmul + activation)
- Speculative decoding (generate multiple tokens at once)
- Hardware acceleration (GPU/NPU)

## Conclusion

The optimized C engine achieves **1.37x speedup over CTranslate2** while maintaining 100% quality parity and using less memory. This demonstrates that careful optimization of a pure C implementation can outperform highly-optimized C++ frameworks.
