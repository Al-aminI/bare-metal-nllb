#!/usr/bin/env python3
"""Compare CT2 and C engine step-by-step to find divergence."""

import ctranslate2
import subprocess
import numpy as np
from transformers import NllbTokenizer

def get_ct2_first_step_logits(source_text="Hello.", src_lang="eng_Latn", tgt_lang="hau_Latn"):
    """Get CT2's logits at the first decoding step."""
    
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = src_lang
    
    # Tokenize
    tokens = tokenizer.encode(source_text, add_special_tokens=True)
    token_strs = tokenizer.convert_ids_to_tokens(tokens)
    
    print(f"Source: {source_text}")
    print(f"Tokens: {tokens}")
    print(f"Token strings: {token_strs}")
    
    # Get CT2 translation with return_logits_vocab
    translator = ctranslate2.Translator(
        "/tmp/nllb-200-600M-ct2-int8",
        device="cpu",
        compute_type="int8"
    )
    
    # Use beam_size=1 for simpler comparison
    results = translator.translate_batch(
        [token_strs],
        target_prefix=[[tgt_lang]],
        beam_size=1,
        max_decoding_length=10,
        length_penalty=0.0,
        return_scores=True,
        return_logits_vocab=True  # This returns logits at each step
    )
    
    print(f"\nCT2 Output: {results[0].hypotheses[0]}")
    print(f"CT2 Score: {results[0].scores[0]}")
    
    # Check if logits are available
    if hasattr(results[0], 'logits_vocab') and results[0].logits_vocab:
        print(f"\nCT2 returned {len(results[0].logits_vocab)} steps of logits")
        
        # First step logits (after tgt_lang token)
        first_step_logits = results[0].logits_vocab[0]
        print(f"First step logits shape: {len(first_step_logits)}")
        
        # Get top 10 tokens
        logits_array = np.array(first_step_logits)
        top_indices = np.argsort(logits_array)[-10:][::-1]
        
        print("\nTop 10 tokens at first step:")
        for idx in top_indices:
            token = tokenizer.convert_ids_to_tokens([idx])[0]
            print(f"  {token:20s} ({idx:6d}): {logits_array[idx]:.4f}")
            
        return logits_array, results[0].hypotheses[0]
    else:
        print("\nCT2 did not return logits (return_logits_vocab may not be supported)")
        return None, results[0].hypotheses[0]

def compare_with_c_engine():
    """Compare CT2 and C engine diagnostics."""
    
    print("=" * 80)
    print("STEP-BY-STEP COMPARISON: CT2 vs C ENGINE")
    print("=" * 80)
    
    # Get CT2 logits
    ct2_logits, ct2_output = get_ct2_first_step_logits()
    
    print("\n" + "=" * 80)
    print("C ENGINE DIAGNOSTICS")
    print("=" * 80)
    
    # Run C engine with same input
    cmd = ["./pico_nllb", "model_int8_ct2.safetensors", "eng_Latn", "hau_Latn", "94124", "248075", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract diagnostic output
    for line in result.stdout.split('\n'):
        if '[diag]' in line or 'rank of' in line:
            print(line)
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    if ct2_logits is not None:
        print("✓ CT2 logits available for comparison")
        print("  Next: Compare specific token logits between CT2 and C engine")
    else:
        print("✗ CT2 logits not available (return_logits_vocab not supported)")
        print("  Will use diagnostic output from C engine to compare")

if __name__ == "__main__":
    compare_with_c_engine()
