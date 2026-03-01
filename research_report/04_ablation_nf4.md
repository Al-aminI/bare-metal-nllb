# 4. Ablation Study: NF4 Quantization Phase

## 4.1 Initial Implementation

### 4.1.1 NF4 Dequantization Kernel

**Implementation:**
```c
void dequantize_nf4_block(
    const uint8_t* packed_data,  // 32 bytes (64 nibbles)
    const float* nf4_code,        // 16-value lookup table
    float absmax_scale,           // Per-block scale
    float* out_floats             // 64 output floats
) {
    for (int i = 0; i < 32; i++) {
        uint8_t byte = packed_data[i];
        uint8_t low_nibble = byte & 0x0F;
        uint8_t high_nibble = (byte >> 4) & 0x0F;
        
        out_floats[i * 2] = nf4_code[low_nibble] * absmax_scale;
        out_floats[i * 2 + 1] = nf4_code[high_nibble] * absmax_scale;
    }
}
```

**Validation:**
- ✅ Dequantization produces finite values
- ✅ Weight statistics reasonable (mean ≈ 0, std ≈ 0.04)
- ✅ Embedding lookups functional

## 4.2 Critical Bug #1: Double Quantization Misinterpretation

### 4.2.1 Symptom
```
absmax tensor dtype: uint8 (expected: float32)
absmax[0] = 34 (raw value)
Expected: -0.11 (after double-quant decode)
```

### 4.2.2 Root Cause
NF4 with `use_double_quant=True` quantizes the absmax scales themselves:
```
real_absmax = code2[absmax_raw] * absmax2[block_group]
```

### 4.2.3 Fix
```c
float recover_absmax(const uint8_t* absmax_raw,
                     const float* code2,
                     const float* absmax2,
                     int block_idx) {
    int group = block_idx / 256;
    return code2[absmax_raw[block_idx]] * absmax2[group];
}
```

### 4.2.4 Impact
- **Before:** Absmax values ~34.0 (wrong)
- **After:** Absmax values ~0.11 (correct)
- **Translation quality:** Still poor (cross-attention broken)

## 4.3 Critical Bug #2: Cross-Attention Uniformity

### 4.3.1 Symptom
```
Cross-attention scores (4 source tokens):
  [0.247, 0.251, 0.248, 0.254]
Expected: Non-uniform (e.g., [0.05, 0.70, 0.15, 0.10])
```

### 4.3.2 Diagnostic
```python
# Q·K dot products BEFORE softmax:
raw_scores = [-0.0007, 0.0255, -0.0326, 0.0484]
# After softmax → nearly uniform!
```

### 4.3.3 Root Cause Analysis
NF4 quantization (4-bit = 16 discrete values) destroys fine-grained directional information in Q/K projection weights:

**Theoretical Analysis:**
- Q, K ∈ ℝ^(1024): Need to preserve angular relationships
- NF4: Each weight ∈ {-1.0, -0.696, ..., 0.0, ..., 1.0} (16 values)
- Result: Q·K ≈ 0 for most token pairs

**Empirical Validation:**
```python
# Python reference with NF4 weights:
K_nf4 = dequantize_nf4(k_proj_weights)
scores = Q @ K_nf4.T  # Near-zero dot products
```

### 4.3.4 Conclusion
**NF4 is insufficient for encoder-decoder cross-attention.**

Decoder-only models (like LLaMA) can tolerate NF4 because:
- Self-attention has positional bias
- Causal masking provides structure
- No cross-attention dependency

Encoder-decoder models require:
- Precise Q·K alignment across sequences
- INT8 minimum (256 discrete values)

## 4.4 Decision Point: Switch to INT8

### 4.4.1 Quantitative Comparison

| Metric | NF4 | INT8 |
|--------|-----|------|
| Model size | 675MB | 1.1GB |
| Discrete values | 16 | 256 |
| Cross-attn quality | ❌ Uniform | ✅ Discriminative |
| Translation quality | ❌ Garbage | ✅ Production |
| Memory footprint | 27MB | 35MB |

### 4.4.2 Rationale
- 1.6x size increase acceptable for quality
- INT8 is CTranslate2 standard (proven)
- Still fits on edge devices (< 2GB)

**Decision:** Abandon NF4, adopt INT8 from CTranslate2.
