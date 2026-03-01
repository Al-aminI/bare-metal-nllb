# 6. Complete Bug Taxonomy

## 6.1 Quantization Bugs

### Bug #1: Double Quantization Misinterpretation
- **Phase:** NF4 implementation
- **Severity:** Critical
- **Symptom:** Absmax values 300x too large
- **Root cause:** Treated uint8 absmax as float32
- **Fix:** Implement two-level dequantization
- **Impact:** Enabled NF4 math (but quality still poor)

### Bug #2: NF4 Code Table Precision
- **Phase:** NF4 implementation
- **Severity:** Minor
- **Symptom:** Small numerical drift (0.000050 per value)
- **Root cause:** Hardcoded truncated values
- **Fix:** Load exact float32 values from safetensors
- **Impact:** Marginal improvement (~0.1% accuracy)

## 6.2 Architecture Bugs

### Bug #3: BOS Token Confusion
- **Phase:** Initial decoder
- **Severity:** Critical
- **Symptom:** Decoder starts with wrong token
- **Root cause:** Used token 2 (EOS) as BOS
- **Fix:** Discovered NLLB uses EOS as decoder_start_token
- **Impact:** Reverted fix (original was correct!)

### Bug #4: Cross-Attention K/V Never Projected
- **Phase:** Initial decoder
- **Severity:** Critical
- **Symptom:** Decoder ignores encoder output
- **Root cause:** `if (step == 0)` never true (started at step=1)
- **Fix:** Start generation at step=0 with decoder_start_token
- **Impact:** Enabled cross-attention

### Bug #5: BOS Never Processed
- **Phase:** Initial decoder
- **Severity:** Critical
- **Symptom:** Position 0 in KV cache all zeros
- **Root cause:** Generation started with n_gen=2 (BOS+lang pre-filled)
- **Fix:** Start with n_gen=1, process BOS at step=0
- **Impact:** Fixed KV cache initialization

## 6.3 Scaling Bugs

### Bug #6: Embedding Scaling Order
- **Phase:** NF4 implementation
- **Severity:** Medium
- **Symptom:** Positional encoding overwhelms embeddings
- **Root cause:** Applied sqrt(d_model) after adding positions
- **Fix:** Scale embeddings THEN add positions
- **Impact:** Improved but not sufficient

### Bug #7: Bias Dtype Mismatch
- **Phase:** INT8 implementation
- **Severity:** Critical
- **Symptom:** All bias values garbage
- **Root cause:** Read float32 biases as uint16_t (fp16)
- **Fix:** Change bias pointer type to `const float*`
- **Impact:** Major quality improvement

### Bug #8: Scale Direction (INT8)
- **Phase:** INT8 implementation
- **Severity:** Critical
- **Symptom:** Logits 70,000x too large
- **Root cause:** Multiplied by scale instead of dividing
- **Fix:** `float = int8 / scale` (not `int8 * scale`)
- **Impact:** Brought logits to reasonable range

### Bug #9: Encoder Embedding Scale Mismatch
- **Phase:** INT8 implementation
- **Severity:** Critical
- **Symptom:** Different translations than CT2
- **Root cause:** Used single scale for encoder, per-row for decoder
- **Fix:** Share per-row scales (embeddings are shared)
- **Impact:** **Achieved exact parity with CT2**

## 6.4 Numerical Bugs

### Bug #10: Sinusoidal Position Encoding
- **Phase:** NF4 implementation
- **Severity:** Medium
- **Symptom:** Repeated token generation
- **Root cause:** Wrong frequency formula and dimension layout
- **Fix:** Use NLLB's M2M100 sinusoidal formula
- **Impact:** Reduced repetition

### Bug #11: Language Token Masking
- **Phase:** NF4 implementation
- **Severity:** Medium
- **Symptom:** Language tokens dominate logits
- **Root cause:** Language token embeddings have larger norms
- **Fix:** Mask language tokens (256000+) after forced step
- **Impact:** Prevented language token loops

### Bug #12: EOS Blocked by Repetition Penalty
- **Phase:** NF4 implementation
- **Severity:** Medium
- **Symptom:** Model never generates EOS
- **Root cause:** BOS and EOS both token 2, penalty blocks EOS
- **Fix:** Exempt EOS from repetition penalty
- **Impact:** Enabled natural termination

### Bug #13: Vocab Projection Scaling (The 32x Bug)
- **Phase:** Production validation (post-INT8)
- **Severity:** Critical
- **Symptom:** Shorter translations, beam search favoring brevity, 2-3x worse scores
- **Root cause:** Applied `1/sqrt(d_model)` scaling during vocab projection
- **Fix:** Remove embedding scale from output projection (only for input)
- **Impact:** **Achieved 100% exact parity with CT2 on all test cases**

#### Detailed Analysis: Bug #13

This bug was discovered during production validation when testing longer sentences. While short sentences (3-4 tokens) showed 60% exact match rate, longer sentences produced systematically shorter and less accurate translations.

**Investigation Process:**

1. **Symptom Recognition:**
   - C engine: "Hanyar kimiyya hanya ce ta koyon duniya." (10 tokens)
   - CT2: "Hanyar kimiyya hanya ce mai kyau na koyo game da duniya." (13 tokens)
   - C engine missing: "mai kyau" (good), "game" (about)

2. **Score Analysis:**
   ```
   Test: "Hello."
   CT2 cumulative score: -17.87 (5 tokens)
   CT2 reported score:   -3.35 (normalized)
   C engine score:       -7.99 (normalized)
   Ratio: 2.4x worse
   ```

3. **Token-Level Debugging:**
   ```
   Token: "Bar" (3937)
   CT2 log-prob:     -1.61
   C engine log-prob: -12.04  (7.5x worse!)
   ```

4. **Raw Logit Inspection:**
   ```
   Before softmax:
   CT2 (expected):    3.0 to 13.0 range
   C engine (actual): 0.1 to 0.4 range  (32x too small!)
   ```

5. **Root Cause Identification:**
   ```c
   // WRONG: vocab_project() in decoder.c
   const float inv_embed_scale = 1.0f / sqrtf((float)D_MODEL);  // 1/32
   logits[v] = l * inv_embed_scale;  // Dividing by 32!
   ```

**Why This Happened:**

The embedding scale (`sqrt(d_model) = 32`) is a standard transformer technique to prevent embeddings from being too small relative to positional encodings. However, this scale should only be applied during:
- ✅ Input embedding lookup (to amplify embeddings)
- ❌ Output vocab projection (WRONG - creates asymmetry)

The correct approach:
```
Input:  embedding * sqrt(d_model) + positional_encoding
Output: embedding^T @ hidden_state  (no scaling)
```

**Impact on Beam Search:**

With logits scaled down by 32x:
- All log-probabilities become more negative
- Longer sequences accumulate more negative scores
- Beam search favors shorter sequences (less negative cumulative score)
- Result: Premature termination, missing content

**The Fix:**

```c
// CORRECT: Remove embedding scale from output
void pico_vocab_project(const PicoModel* m, const float* normed,
                        float* logits, int mask_lang) {
    for (int v = 0; v < VOCAB_SIZE; v++) {
        const int8_t* row = m->decoder_embed_weight + (size_t)v * D_MODEL;
        const float inv_s = 1.0f / m->decoder_embed_scale[v];
        float l = 0.0f;
        for (int d = 0; d < D_MODEL; d++)
            l += (float)row[d] * inv_s * normed[d];
        logits[v] = l;  // No embedding scale!
    }
}
```

**Results After Fix:**

| Test Case | Before | After |
|-----------|--------|-------|
| Hello. | ✅ Match (wrong score) | ✅ EXACT MATCH |
| Good morning. | ❌ Different | ✅ EXACT MATCH |
| How are you? | ✅ Match | ✅ EXACT MATCH |
| Thank you. | ✅ Match | ✅ EXACT MATCH |
| Scientific method (long) | ❌ 10 tokens (short) | ✅ 13 tokens EXACT MATCH |

**Overall: 60% → 100% exact parity**

**Key Insight:**

This bug demonstrates that even after achieving "good" results (60% match), subtle numerical issues can prevent production-quality performance. The bug only became apparent when:
1. Testing longer sequences (>10 tokens)
2. Comparing token-level scores (not just final output)
3. Inspecting raw logit magnitudes

## 6.5 Bug Impact Analysis

### 6.5.1 By Severity

| Severity | Count | Examples |
|----------|-------|----------|
| Critical | 8 | Scale direction, shared embeddings, vocab projection (32x bug) |
| Medium | 4 | Sinusoidal encoding, language masking |
| Minor | 1 | Code table precision |

### 6.5.2 By Phase

| Phase | Bugs | Resolution |
|-------|------|------------|
| NF4 | 8 | Partial (architecture fixed, quantization insufficient) |
| INT8 | 4 | Complete (exact parity achieved) |
| Production | 1 | Complete (100% parity achieved) |

**Note:** The NF4 phase had lower memory usage (~27MB runtime) due to smaller quantization, but was unusable for translation. INT8 requires ~128MB runtime but delivers production quality.

### 6.5.3 Root Cause Categories

| Category | Count | Description |
|----------|-------|-------------|
| Quantization semantics | 3 | Scale direction, double-quant, dtype |
| Architecture logic | 3 | BOS processing, cross-attention, KV cache |
| Numerical precision | 4 | Sinusoidal, code table, scaling order, vocab projection |
| Model-specific quirks | 3 | NLLB decoder_start_token, shared embeddings, language tokens |

## 6.6 Lessons Learned

### 6.6.1 Quantization
1. **Always verify scale semantics** (multiply vs divide)
2. **Check tensor dtypes explicitly** (don't assume)
3. **Validate dequantization with reference** (Python comparison)

### 6.6.2 Architecture
1. **Trace execution step-by-step** (print intermediate values)
2. **Verify special tokens** (BOS/EOS/PAD may differ from expectations)
3. **Check shared parameters** (embeddings, layer norms)

### 6.6.3 Debugging Strategy
1. **Start with Python reference** (ground truth)
2. **Compare intermediate values** (not just final output)
3. **Isolate components** (test encoder/decoder separately)
4. **Use diagnostic prints** (logit distributions, attention scores)
5. **Test with varying lengths** (short sentences may hide scaling bugs)
6. **Inspect raw magnitudes** (logits should be -10 to +10, not 0.1 to 0.4)
7. **Compare token-level scores** (cumulative differences reveal systematic issues)

### 6.6.4 Production Validation Insights

The vocab projection bug (Bug #13) was discovered only after achieving "good" results (60% exact match). This highlights critical lessons:

1. **Good ≠ Production-Ready:**
   - 60% exact match seemed acceptable
   - But longer sequences revealed systematic bias
   - Production requires 100% parity on diverse inputs

2. **Test Diverse Lengths:**
   - Short sentences (3-5 tokens): Masked the bug
   - Long sentences (10+ tokens): Exposed the issue
   - Always test edge cases (very short, very long)

3. **Score Magnitude Matters:**
   - Relative rankings can be correct even with wrong magnitudes
   - Absolute score values affect beam search dynamics
   - Compare raw logits, not just final predictions

4. **Beam Search Amplifies Errors:**
   - Small per-token errors compound over sequences
   - Beam search selection is sensitive to score magnitudes
   - Length bias emerges from cumulative score differences

## 6.7 Complete Bug Timeline

```
Day 1-3:   NF4 Implementation
           ├─ Bug #1: Double quantization (FIXED)
           ├─ Bug #2: Code table precision (FIXED)
           ├─ Bug #6: Embedding scaling order (FIXED)
           ├─ Bug #10: Sinusoidal encoding (FIXED)
           ├─ Bug #11: Language token masking (FIXED)
           └─ Bug #12: EOS repetition penalty (FIXED)
           Result: NF4 insufficient for cross-attention

Day 4-6:   INT8 Migration
           ├─ Bug #7: Bias dtype mismatch (FIXED)
           ├─ Bug #8: Scale direction (FIXED)
           └─ Bug #9: Shared embedding scales (FIXED)
           Result: 60% exact parity achieved

Day 7-8:   Architecture Fixes
           ├─ Bug #3: BOS token confusion (REVERTED)
           ├─ Bug #4: Cross-attention never projected (FIXED)
           └─ Bug #5: BOS never processed (FIXED)
           Result: Stable inference, ready for validation

Day 9-10:  Production Validation
           └─ Bug #13: Vocab projection scaling (FIXED)
           Result: 100% exact parity achieved

Total: 13 bugs, 10 days, 100% parity
```
