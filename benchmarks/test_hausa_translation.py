#!/usr/bin/env python3
"""Test English to Hausa translation: CT2 vs C Engine."""

import ctranslate2
import subprocess
from transformers import NllbTokenizer

def test_translation(source_text, src_lang="eng_Latn", tgt_lang="hau_Latn"):
    """Test a single translation and compare CT2 vs C engine."""
    
    print("=" * 80)
    print(f"TEST: '{source_text}'")
    print(f"      {src_lang} → {tgt_lang}")
    print("=" * 80)
    
    # Load tokenizer
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = src_lang
    
    # Tokenize
    tokens = tokenizer.encode(source_text, add_special_tokens=True)
    token_strs = tokenizer.convert_ids_to_tokens(tokens)
    
    print(f"\nSource tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    
    # ─── CT2 Translation ───────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("CT2 REFERENCE (INT8)")
    print("─" * 80)
    
    translator = ctranslate2.Translator(
        "/tmp/nllb-200-600M-ct2-int8",
        device="cpu",
        compute_type="int8"
    )
    
    results = translator.translate_batch(
        [token_strs],
        target_prefix=[[tgt_lang]],
        beam_size=4,
        max_decoding_length=50,
        length_penalty=0.0,  # No length normalization (matches C engine LENGTH_PENALTY=0.0)
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        return_scores=True
    )
    
    ct2_tokens = tokenizer.convert_tokens_to_ids(results[0].hypotheses[0])
    ct2_text = tokenizer.decode(ct2_tokens, skip_special_tokens=True)
    ct2_score = results[0].scores[0]
    
    print(f"Output tokens: {ct2_tokens}")
    print(f"Output text: {ct2_text}")
    print(f"Score: {ct2_score:.4f}")
    print(f"Length: {len(ct2_tokens)} tokens")
    
    # ─── C Engine Translation ──────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("C ENGINE (INT8)")
    print("─" * 80)
    
    # Get language token IDs
    src_token = tokenizer.convert_tokens_to_ids(src_lang)
    tgt_token = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    # Remove the source language token from input (C engine adds it)
    input_tokens = [t for t in tokens if t != src_token]
    
    # Run C engine
    cmd = [
        "./pico_nllb_opt",  # Back to optimized version
        "model_int8_ct2.safetensors",
        src_lang,
        tgt_lang
    ] + [str(t) for t in input_tokens]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd="/Users/mac/Desktop/codes/bare-metal-engine"
        )
        
        # Parse output
        output_lines = result.stdout.strip().split('\n')
        
        # Find the tokens line
        c_tokens = None
        c_score = None
        for line in output_lines:
            if line.startswith("best:"):
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.startswith("score="):
                        c_score = float(part.split('=')[1].rstrip(','))
            if line.startswith("tokens:"):
                c_tokens = [int(t) for t in line.split()[1:]]
                break
        
        if c_tokens:
            c_text = tokenizer.decode(c_tokens, skip_special_tokens=True)
            print(f"Output tokens: {c_tokens}")
            print(f"Output text: {c_text}")
            if c_score is not None:
                print(f"Score: {c_score:.4f}")
            print(f"Length: {len(c_tokens)} tokens")
            
            # ─── Comparison ────────────────────────────────────────────────────
            print("\n" + "=" * 80)
            print("COMPARISON")
            print("=" * 80)
            
            # Token-level comparison
            match_count = 0
            min_len = min(len(ct2_tokens), len(c_tokens))
            for i in range(min_len):
                if ct2_tokens[i] == c_tokens[i]:
                    match_count += 1
            
            token_accuracy = (match_count / max(len(ct2_tokens), len(c_tokens))) * 100
            
            print(f"CT2 text:  {ct2_text}")
            print(f"C text:    {c_text}")
            print(f"\nCT2 tokens (excluding lang prefix): {[t for t in ct2_tokens if t != 256066]}")
            print(f"C tokens (excluding EOS):            {[t for t in c_tokens if t != 2]}")
            print(f"\nToken accuracy: {token_accuracy:.1f}% ({match_count}/{max(len(ct2_tokens), len(c_tokens))} tokens match)")
            print(f"Length diff: {len(c_tokens) - len(ct2_tokens)} tokens")
            print(f"Score ratio: C/CT2 = {c_score/ct2_score:.1f}x")
            
            if ct2_text.lower() == c_text.lower():
                print("✅ EXACT MATCH (case-insensitive)")
            elif ct2_text == c_text:
                print("✅ EXACT MATCH")
            elif token_accuracy >= 80:
                print("⚠️  CLOSE MATCH")
            else:
                print("❌ MISMATCH")
                
        else:
            print("❌ Failed to parse C engine output")
            print("STDOUT:", result.stdout[-500:])
            print("STDERR:", result.stderr[-500:] if result.stderr else "")
            
    except subprocess.TimeoutExpired:
        print("❌ C engine timed out")
    except Exception as e:
        print(f"❌ Error running C engine: {e}")
    
    print()

def main():
    """Run Hausa translation test suite."""
    
    print("\n" + "=" * 80)
    print("ENGLISH → HAUSA TRANSLATION TEST")
    print("CT2 vs C ENGINE COMPARISON")
    print("=" * 80)
    print()
    
    # Test cases
    test_cases = [
        "Hello.",
        "Good morning.",
        "How are you?",
        "Thank you very much.",
        "The scientific method is a systematic way of learning about the world.",
    ]
    
    for source_text in test_cases:
        test_translation(source_text, "eng_Latn", "hau_Latn")
    
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
