#!/usr/bin/env python3
"""Direct benchmark: measure just translation time, not subprocess overhead."""

import ctranslate2
import time
from transformers import NllbTokenizer

def main():
    print("=" * 80)
    print("DIRECT BENCHMARK: CTranslate2 (pure translation time)")
    print("=" * 80)
    print()
    
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = "eng_Latn"
    
    translator = ctranslate2.Translator('/tmp/nllb-200-600M-ct2-int8', device='cpu', compute_type='int8')
    
    test_cases = [
        ("Short", "Hello."),
        ("Medium", "Thank you very much."),
        ("Long", "The scientific method is a systematic way of learning about the world.")
    ]
    
    print(f"{'Test':<10} {'Time (ms)':<12} {'Tokens':<8} {'tok/s':<10}")
    print("-" * 80)
    
    for name, text in test_cases:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        token_strs = tokenizer.convert_ids_to_tokens(tokens)
        
        # Warmup
        translator.translate_batch(
            [token_strs],
            target_prefix=[["hau_Latn"]],
            beam_size=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            length_penalty=0.0
        )
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            result = translator.translate_batch(
                [token_strs],
                target_prefix=[["hau_Latn"]],
                beam_size=4,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                length_penalty=0.0
            )
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = sum(times) / len(times)
        n_tokens = len(result[0].hypotheses[0]) - 1  # Exclude lang token
        tps = n_tokens / avg_time
        
        print(f"{name:<10} {avg_time*1000:>8.1f}ms    {n_tokens:<8} {tps:>6.2f}")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
