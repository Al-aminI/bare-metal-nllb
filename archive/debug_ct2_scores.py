#!/usr/bin/env python3
"""Debug CT2 scoring to understand the 2-3x score difference."""

import ctranslate2
from transformers import NllbTokenizer
import subprocess

def test_scoring(source_text, src_lang="eng_Latn", tgt_lang="hau_Latn"):
    """Compare CT2 and C engine scores in detail."""
    
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = src_lang
    
    tokens = tokenizer.encode(source_text, add_special_tokens=True)
    token_strs = tokenizer.convert_ids_to_tokens(tokens)
    
    print("=" * 80)
    print(f"Testing: {source_text}")
    print(f"Tokens: {tokens}")
    print("=" * 80)
    
    # CT2 translation
    translator = ctranslate2.Translator(
        "/tmp/nllb-200-600M-ct2-int8",
        device="cpu",
        compute_type="int8"
    )
    
    results = translator.translate_batch(
        [token_strs],
        target_prefix=[[tgt_lang]],
        beam_size=1,  # Greedy for simpler comparison
        max_decoding_length=20,
        length_penalty=0.0,
        return_scores=True
    )
    
    ct2_tokens = tokenizer.convert_tokens_to_ids(results[0].hypotheses[0])
    ct2_text = tokenizer.decode(ct2_tokens, skip_special_tokens=True)
    ct2_score = results[0].scores[0]
    
    print(f"\nCT2 (beam=1, greedy):")
    print(f"  Tokens: {ct2_tokens}")
    print(f"  Text: {ct2_text}")
    print(f"  Score: {ct2_score:.4f}")
    print(f"  Length: {len(ct2_tokens)} tokens")
    
    # Now score the same sequence with score_batch
    target_tokens = results[0].hypotheses[0]
    scores = translator.score_batch([token_strs], [target_tokens])
    scoring_result = scores[0]
    
    print(f"\nCT2 score_batch result:")
    print(f"  Type: {type(scoring_result)}")
    print(f"  Dir: {[x for x in dir(scoring_result) if not x.startswith('_')]}")
    
    # Try to get token scores
    if hasattr(scoring_result, 'tokens_score'):
        token_scores = scoring_result.tokens_score
    elif hasattr(scoring_result, 'log_probs'):
        token_scores = scoring_result.log_probs
    else:
        print("  Cannot find token-level scores")
        token_scores = None
    
    if token_scores:
        print(f"\nCT2 score_batch (token-level scores):")
        for i, (tok, score) in enumerate(zip(target_tokens, token_scores)):
            tok_id = tokenizer.convert_tokens_to_ids([tok])[0]
            print(f"  Step {i}: {tok:20s} ({tok_id:6d}) score={score:.4f}")
        
        cumulative = sum(token_scores)
        average = cumulative / len(token_scores)
        print(f"\n  Cumulative: {cumulative:.4f}")
        print(f"  Average: {average:.4f}")
    
    if hasattr(scoring_result, 'score'):
        print(f"  Score from scoring_result: {scoring_result.score:.4f}")
    print(f"  translate_batch score: {ct2_score:.4f}")
    
    # C engine
    print(f"\n" + "-" * 80)
    print("C ENGINE:")
    print("-" * 80)
    
    src_token = tokenizer.convert_tokens_to_ids(src_lang)
    input_tokens = [t for t in tokens if t != src_token]
    
    cmd = ["./pico_nllb", "model_int8_ct2.safetensors", src_lang, tgt_lang] + [str(t) for t in input_tokens]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse C engine output
    for line in result.stdout.split('\n'):
        if 'best:' in line or 'tokens:' in line:
            print(line)
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    # Test with short sentence
    test_scoring("Hello.", "eng_Latn", "hau_Latn")
    
    print("\n\n")
    
    # Test with longer sentence
    test_scoring("How are you?", "eng_Latn", "hau_Latn")
