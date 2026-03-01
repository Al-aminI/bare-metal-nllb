#!/usr/bin/env python3
"""Benchmark performance: baseline vs optimized C engine vs CTranslate2."""

import subprocess
import time
from transformers import NllbTokenizer
import ctranslate2

def benchmark_engine(binary, test_text, n_runs=3):
    """Benchmark a single engine."""
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = "eng_Latn"
    
    tokens = tokenizer.encode(test_text, add_special_tokens=True)
    src_token = tokenizer.convert_tokens_to_ids("eng_Latn")
    input_tokens = [t for t in tokens if t != src_token]
    
    cmd = [
        f"./{binary}",
        "model_int8_ct2.safetensors",
        "eng_Latn",
        "hau_Latn"
    ] + [str(t) for t in input_tokens]
    
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            times.append(elapsed)
        else:
            print(f"Error running {binary}")
            return None
    
    return min(times)  # Best of n_runs

def benchmark_ct2(test_text, n_runs=5):
    """Benchmark CTranslate2."""
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer.src_lang = "eng_Latn"
    
    tokens = tokenizer.encode(test_text, add_special_tokens=True)
    token_strs = tokenizer.convert_ids_to_tokens(tokens)
    
    translator = ctranslate2.Translator('/tmp/nllb-200-600M-ct2-int8', device='cpu', compute_type='int8')
    
    # Warmup
    translator.translate_batch(
        [token_strs],
        target_prefix=[["hau_Latn"]],
        beam_size=4,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        length_penalty=0.0
    )
    
    times = []
    for _ in range(n_runs):
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
    
    return min(times)  # Best of n_runs

def main():
    print("=" * 80)
    print("PERFORMANCE BENCHMARK: Baseline vs Optimized vs CTranslate2")
    print("=" * 80)
    print()
    
    test_cases = [
        ("Short", "Hello."),
        ("Medium", "Thank you very much."),
        ("Long", "The scientific method is a systematic way of learning about the world.")
    ]
    
    print(f"{'Test':<10} {'Baseline':<12} {'Optimized':<12} {'CT2':<12} {'vs Base':<10} {'vs CT2':<10}")
    print("-" * 80)
    
    for name, text in test_cases:
        baseline_time = benchmark_engine("pico_nllb_baseline", text, n_runs=5)
        opt_time = benchmark_engine("pico_nllb_opt", text, n_runs=5)
        ct2_time = benchmark_ct2(text, n_runs=5)
        
        if baseline_time and opt_time and ct2_time:
            speedup_base = baseline_time / opt_time
            speedup_ct2 = ct2_time / opt_time
            print(f"{name:<10} {baseline_time:>8.3f}s    {opt_time:>8.3f}s    {ct2_time:>8.3f}s    {speedup_base:>6.2f}x    {speedup_ct2:>6.2f}x")
        else:
            print(f"{name:<10} ERROR")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
