# 9. Discussion

## 9.1 Key Findings

### 9.1.1 Quantization for Encoder-Decoder Models

**Finding 1: NF4 is insufficient for cross-attention.**

Our experiments definitively show that 4-bit quantization destroys the fine-grained directional information required for encoder-decoder cross-attention:

```
NF4 cross-attention scores: [0.247, 0.251, 0.248, 0.254]  (nearly uniform)
INT8 cross-attention scores: [0.05, 0.70, 0.15, 0.10]     (discriminative)
```

**Theoretical explanation:**
- Cross-attention requires Q·K to discriminate between source positions
- NF4 provides only 16 discrete values per weight
- This quantization noise overwhelms the signal in Q·K dot products
- Result: Attention becomes nearly uniform, decoder ignores encoder

**Implication:** Decoder-only models (LLaMA, GPT) can use NF4 because:
- Self-attention has positional bias and causal masking
- No cross-attention dependency
- Larger models (7B+) have more redundancy

Encoder-decoder models require INT8 minimum.

### 9.1.2 Shared Embedding Quantization

**Finding 2: Shared embeddings must use unified quantization scales.**

NLLB shares embedding weights between encoder and decoder:
```python
encoder_weights == decoder_weights  # Same int8 values
```

**Critical insight:** The quantization scales must also be shared:
```c
// WRONG:
encoder: weights / 32.0           (single global scale)
decoder: weights / scales[token]  (per-row scales)

// CORRECT:
encoder: weights / scales[token]  (per-row scales)
decoder: weights / scales[token]  (per-row scales)
```

**Why this matters:**
- Encoder and decoder must produce compatible representations
- Different scales → different embedding spaces → broken cross-attention
- This bug caused 40% of our translation errors

**Generalization:** Any model with shared parameters must use unified quantization.

### 9.1.3 Scale Semantics in Quantization

**Finding 3: Quantization scale direction is framework-specific.**

Different frameworks use different conventions:

| Framework | Scale Semantics | Dequantization |
|-----------|----------------|----------------|
| bitsandbytes | Dequant scale | `float = int8 * scale` |
| CTranslate2 | Quant scale | `float = int8 / scale` |
| GGML | Dequant scale | `float = int8 * scale` |

**Lesson:** Always validate scale direction with reference implementation.

## 9.2 Comparison with Related Work

### 9.2.1 vs CTranslate2

**Advantages of our approach:**
- Simpler codebase (~2,500 lines vs ~50,000)
- No external dependencies (vs Boost, Intel MKL)
- Easier to understand and modify
- Slightly lower memory (130MB vs 150MB)

**Advantages of CTranslate2:**
- 1.5-2x faster (SIMD, threading)
- More features (batching, sampling, multiple models)
- Production-tested (used by major companies)
- Active maintenance

**Conclusion:** Our implementation is ideal for:
- Learning/research
- Embedded systems (simplicity matters)
- Custom modifications

CTranslate2 is better for:
- Production deployments
- High-throughput scenarios
- Standard use cases

### 9.2.2 vs llama.cpp

**Similarities:**
- Bare-metal C/C++ implementation
- Quantization-focused
- Minimal dependencies
- Community-driven

**Differences:**
- llama.cpp: Decoder-only (LLaMA, GPT)
- Our work: Encoder-decoder (NLLB, mBART)
- llama.cpp: Extensive SIMD optimization
- Our work: Portable C11

**Contribution:** We extend the llama.cpp philosophy to encoder-decoder models.

### 9.2.3 vs ONNX Runtime

**Advantages of ONNX:**
- Broad model support
- Multiple backends (CPU, GPU, NPU)
- Automatic optimization

**Advantages of our approach:**
- No runtime dependencies
- Smaller binary (~100KB vs ~50MB)
- Predictable performance
- Full control over execution

## 9.3 Practical Implications

### 9.3.1 For Researchers

**Reproducibility:**
- Complete bug taxonomy enables others to avoid our mistakes
- Validation methodology (token-level comparison) is reusable
- Ablation study format is applicable to other models

**Extensibility:**
- Codebase is small enough to understand fully
- Adding new features is straightforward
- Good foundation for research prototypes

### 9.3.2 For Practitioners

**Deployment scenarios:**
1. **Edge devices:** Raspberry Pi, Jetson Nano
2. **Embedded systems:** Industrial controllers, IoT gateways
3. **Air-gapped systems:** Secure environments without internet
4. **Cost optimization:** Avoid GPU costs for low-volume use

**Integration:**
```c
// Simple API:
PicoModel model;
pico_load(&model, "model.safetensors");

float* encoder_out = pico_encode(&model, tokens, n_tokens);
int* translation = pico_translate(&model, encoder_out, n_tokens, tgt_lang);
```

### 9.3.3 For Educators

**Teaching value:**
- Demonstrates complete ML system (data → deployment)
- Shows real-world debugging process
- Illustrates quantization trade-offs
- Provides hands-on transformer implementation

**Course integration:**
- ML systems course: Quantization and deployment
- Compilers course: Optimization techniques
- Software engineering: Debugging methodology

## 9.4 Threats to Validity

### 9.4.1 Internal Validity

**Test suite size:**
- Only 5 test cases
- May not cover all edge cases
- Mitigation: Tests span different lengths and complexities

**Reference implementation:**
- CTranslate2 as ground truth (not original PyTorch)
- Assumes CT2 is correct
- Mitigation: CT2 is industry-standard, well-tested

### 9.4.2 External Validity

**Single model:**
- Only tested NLLB-200-distilled-600M
- May not generalize to other sizes/architectures
- Mitigation: Architecture is standard transformer

**Single language pair:**
- Primarily English → French
- Other language pairs may behave differently
- Mitigation: NLLB is multilingual by design

**Single hardware:**
- Tested on Apple M1 only
- Performance may differ on other CPUs
- Mitigation: Used portable C11, no platform-specific code

### 9.4.3 Construct Validity

**Translation quality:**
- Measured by token-level accuracy
- Doesn't capture semantic equivalence fully
- Mitigation: Manual inspection of outputs

**Performance:**
- Measured on single-threaded, unoptimized code
- Not representative of optimized implementation
- Mitigation: Clearly documented optimization opportunities

## 9.5 Lessons for Future Work

### 9.5.1 Quantization Research

1. **Test cross-attention explicitly** when evaluating quantization schemes
2. **Validate shared parameters** have unified quantization
3. **Compare intermediate values**, not just final outputs
4. **Document scale semantics** clearly in papers

### 9.5.2 Systems Implementation

1. **Start with reference implementation** (Python/PyTorch)
2. **Validate each component** before integration
3. **Use diagnostic prints** liberally during development
4. **Test with multiple inputs** to catch caching bugs

### 9.5.3 Open Source

1. **Document bugs and fixes** (helps others)
2. **Provide validation scripts** (enables reproduction)
3. **Include performance baselines** (sets expectations)
4. **Write clear README** (lowers barrier to entry)
