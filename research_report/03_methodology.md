# 3. Methodology

## 3.1 Experimental Setup

### 3.1.1 Hardware
- **CPU:** Apple M1 (x86_64 emulation)
- **RAM:** 16GB
- **OS:** macOS (darwin)
- **Compiler:** clang 14.0 with `-O2 -ffast-math`

### 3.1.2 Software Stack
- **Reference:** CTranslate2 4.7.1 (INT8)
- **Tokenizer:** HuggingFace Transformers 5.2.0
- **Quantization:** bitsandbytes 0.44.1 (NF4), CTranslate2 (INT8)
- **Validation:** Python 3.13 + NumPy 2.4.2

### 3.1.3 Model Variants
1. **Original:** facebook/nllb-200-distilled-600M (FP32, 2.4GB)
2. **NF4:** Custom quantization (675MB)
3. **INT8:** CTranslate2 quantization (1.1GB)

## 3.2 Quantization Pipeline

### 3.2.1 NF4 Quantization (Initial Attempt)

**Process:**
```python
from transformers import AutoModelForSeq2SeqLM
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
```

**Output:**
- Model size: 675MB
- Quantization scheme: NF4 with double quantization
- Block size: 64 elements
- Per-block absmax scales

### 3.2.2 INT8 Quantization (Final)

**Process:**
```bash
ct2-transformers-converter \
    --model facebook/nllb-200-distilled-600M \
    --output_dir nllb-200-600M-ct2-int8 \
    --quantization int8
```

**Output:**
- Model size: 1.1GB
- Quantization scheme: Per-row INT8 scales
- Shared embeddings: Encoder and decoder use same weights

## 3.3 Implementation Phases

### Phase 1: NF4 Implementation (Days 1-3)
- Custom safetensors format
- NF4 dequantization kernels
- Initial encoder/decoder implementation

### Phase 2: Bug Discovery (Days 4-7)
- Cross-attention uniformity detected
- NF4 precision issues identified
- Decision to switch to INT8

### Phase 3: INT8 Migration (Days 8-10)
- CTranslate2 model conversion
- INT8 dequantization implementation
- Scale direction bug discovery

### Phase 4: Validation (Days 11-12)
- Token-level comparison framework
- Shared embedding bug discovery
- Final parity achievement

## 3.4 Validation Methodology

### 3.4.1 Test Suite
Five representative test cases:
1. **Short greeting:** "Hello." (4 tokens)
2. **Polite greeting:** "Good morning." (5 tokens)
3. **Technical text:** "The scientific method..." (16 tokens)
4. **Question:** "How are you?" (6 tokens)
5. **Gratitude:** "Thank you very much." (7 tokens)

### 3.4.2 Metrics
- **Token accuracy:** Exact match percentage
- **Semantic equivalence:** Human evaluation
- **Latency:** Time per token (tok/s)
- **Memory:** Peak RSS during inference

### 3.4.3 Comparison Protocol
```python
# For each test case:
1. Tokenize with NLLB tokenizer
2. Run CTranslate2 (reference)
3. Run C engine (test)
4. Compare token-by-token
5. Decode and compare text
```
