# 7. System Architecture

## 7.1 Overview

The final system consists of 8 C files (~2,500 lines) implementing a complete NLLB inference engine:

```
pico_nllb/
├── pico.h           (350 lines)  - Shared types and constants
├── loader.c         (400 lines)  - Model loading and memory mapping
├── tensor.c         (200 lines)  - INT8 operations and math primitives
├── encoder.c        (300 lines)  - 12-layer transformer encoder
├── decoder.c        (400 lines)  - 12-layer transformer decoder
├── main.c           (600 lines)  - Beam search and CLI
├── Makefile         (50 lines)   - Build system
└── verify_before_c.py (200 lines) - Python validation
```

## 7.2 Memory Architecture

### 7.2.1 Model Storage (mmap)
```c
// Zero-copy model loading
int fd = open("model_int8_ct2.safetensors", O_RDONLY);
void* mmap_ptr = mmap(NULL, 1.1GB, PROT_READ, MAP_PRIVATE, fd, 0);

// Direct pointer access (no heap allocation for weights)
const int8_t* weights = (const int8_t*)(mmap_ptr + offset);
```

**Benefits:**
- OS handles paging (only active layers in RAM)
- No heap allocation for 1.1GB model
- Instant "loading" (just map virtual memory)

### 7.2.2 Runtime Buffers

| Buffer | Size | Purpose |
|--------|------|---------|
| Encoder output | n_src × 1024 × 4B | Cached encoder states |
| KV cache (per beam) | 12 × 2 × 256 × 1024 × 4B = 24MB | Decoder self-attention |
| Cross-attn cache | 12 × 2 × 256 × 1024 × 4B = 24MB | Shared across beams |
| Logits buffer | 256K × 4B = 1MB | Vocab projection |
| Scratch buffers | ~5MB | Temporary computations |

**Total runtime memory:** ~35MB (for 4 beams, 256 max length)

### 7.2.3 Memory Access Pattern

```
Encoder (sequential):
  Layer 0 → Layer 1 → ... → Layer 11 → Evict

Decoder (per-step):
  Layer 0 → Layer 1 → ... → Layer 11 → Evict
  (Repeat for each generated token)
```

**Peak memory:** ~35MB (OS pages in ~2MB per layer on-demand)

## 7.3 Computational Pipeline

### 7.3.1 Encoder Forward Pass

```c
void pico_encode(PicoModel* m, const int* tokens, int n, float* out) {
    // 1. Embedding lookup (INT8 → FP32)
    for (int t = 0; t < n; t++) {
        encoder_embed_lookup(m->encoder_embed_weight,
                            m->encoder_embed_scale_array,
                            tokens[t], out + t * 1024);
        
        // 2. Scale by sqrt(d_model)
        for (int i = 0; i < 1024; i++)
            out[t * 1024 + i] *= 32.0;
        
        // 3. Add sinusoidal positions
        add_sinusoidal_pos(out + t * 1024, t, 1024);
    }
    
    // 4. 12 transformer layers
    for (int l = 0; l < 12; l++)
        encoder_layer_forward(&m->encoder_layers[l], n, out);
    
    // 5. Final layer norm
    for (int t = 0; t < n; t++)
        layernorm(out + t * 1024, ...);
}
```

**Complexity:** O(n² × d) for self-attention, O(n × d²) for FFN

### 7.3.2 Decoder Forward Pass (Single Step)

```c
void pico_decode_forward(PicoModel* m, const float* enc_out,
                         int n_src, int cur_token, int step,
                         float* kv_cache, float* xattn_cache,
                         float* out_normed) {
    // 1. Embedding + scale + position
    decoder_embed_lookup(...);
    scale_by_sqrt_d_model(...);
    add_sinusoidal_pos(...);
    
    // 2. 12 decoder layers
    for (int l = 0; l < 12; l++) {
        // Self-attention (causal, uses KV cache)
        decoder_self_attn_cached(...);
        
        // Cross-attention (attends to encoder output)
        decoder_cross_attn_cached(...);
        
        // FFN
        ffn_forward(...);
    }
    
    // 3. Final layer norm
    layernorm(out_normed, ...);
}
```

**Complexity:** O(step × d) for self-attention, O(n_src × d) for cross-attention

### 7.3.3 Beam Search

```c
// Initialize with decoder_start_token (EOS = 2)
beams[0].tokens = [2];

// Step 0: Process decoder_start_token, project cross-attention
pico_decode_forward(m, enc_out, n_src, 2, 0, kv_cache, xattn_cache, normed);

// Force target language token
for (int b = 0; b < BEAM_SIZE; b++)
    beams[b].tokens[1] = tgt_lang_token;

// Step 1+: Generate content tokens
for (int step = 1; step < max_len; step++) {
    // Expand each beam
    for (int b = 0; b < BEAM_SIZE; b++) {
        pico_decode_forward(...);
        pico_vocab_project(...);
        apply_repetition_penalty(...);
        apply_no_repeat_ngram(...);
        find_topk(...);
    }
    
    // Select top BEAM_SIZE candidates
    sort_and_prune(...);
    
    // Early exit if top beam finished
    if (beams[0].finished) break;
}
```

## 7.4 INT8 Quantization Details

### 7.4.1 Per-Row Quantization

```c
void mat_vec_int8(const LinearInt8* layer,
                  const float* x, float* out) {
    for (int row = 0; row < out_features; row++) {
        // 1. Compute int8 dot product
        int32_t acc = 0;
        for (int col = 0; col < in_features; col++)
            acc += (int32_t)weight[row][col] * (int32_t)x[col];
        
        // 2. Dequantize with per-row scale
        float y = (float)acc / scale[row];
        
        // 3. Add bias
        out[row] = y + bias[row];
    }
}
```

**Key insight:** Accumulate in int32, dequantize once per row.

### 7.4.2 Embedding Quantization

```c
// Shared embeddings (encoder and decoder use same weights)
const int8_t* shared_weights = ...;  // [256206, 1024]
const float* shared_scales = ...;    // [256206]

// Encoder lookup
void encoder_embed_lookup(int token_id, float* out) {
    const int8_t* row = shared_weights + token_id * 1024;
    float scale = shared_scales[token_id];
    
    for (int i = 0; i < 1024; i++)
        out[i] = (float)row[i] / scale;
}

// Decoder lookup (identical)
void decoder_embed_lookup(int token_id, float* out) {
    // Same implementation
}
```

**Critical:** Both use the SAME per-row scales.

## 7.5 Optimization Techniques

### 7.5.1 Compiler Flags
```makefile
CFLAGS = -O2 -std=c11 -Wall -Wextra -ffast-math -DNDEBUG
```

- `-O2`: Aggressive optimization
- `-ffast-math`: Relaxed IEEE 754 (safe for NMT)
- `-DNDEBUG`: Remove assertions

### 7.5.2 Memory Layout
- **Contiguous buffers:** Minimize cache misses
- **Aligned allocations:** SIMD-friendly (future work)
- **Reuse scratch space:** Reduce allocations

### 7.5.3 Algorithmic
- **KV caching:** O(n) → O(1) for decoder self-attention
- **Cross-attention caching:** Project encoder K/V once
- **Early exit:** Stop when top beam finishes (length_penalty=0)

## 7.6 Portability

### 7.6.1 Platform Support
- **macOS:** Primary development (x86_64, ARM64)
- **Linux:** Tested (x86_64, ARM)
- **Bare-metal:** Requires `mmap` or `fread` fallback

### 7.6.2 Dependencies
- **Required:** C11 compiler, libc, libm
- **Optional:** None (no external libraries)

### 7.6.3 Build Targets
```makefile
make          # Host platform
make arm      # Cross-compile for ARM
make debug    # Debug build with symbols
```
