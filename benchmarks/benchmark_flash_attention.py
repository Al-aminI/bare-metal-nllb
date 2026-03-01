#!/usr/bin/env python3
"""Benchmark Flash Attention vs Baseline."""

import subprocess
import time
from transformers import NllbTokenizer

def benchmark_engine(binary, test_text, n_runs=5):
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
    enc_times = []
    dec_times = []
    
    for _ in range(n_runs):
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Parse timing from output
            for line in result.stdout.split('\n'):
                if 'best:' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.startswith('enc='):
                            enc_ms = int(part.split('=')[1].rstrip('ms,'))
                            enc_times.append(enc_ms)
                        elif part.startswith('dec='):
                            dec_ms = int(part.split('=')[1].rstrip('ms,'))
                            dec_times.append(dec_ms)
                    times.append(enc_ms + dec_ms)
                    break
        else:
            print(f"Error running {binary}")
            return None, None, None
    
    if times:
        return min(times), min(enc_times), min(dec_times)
    return None, None, None

def main():
    print("=" * 80)
    print("FLASH ATTENTION BENCHMARK")
    print("=" * 80)
    print()
    
    test_cases = [
        ("Short (4 tokens)", "Hello."),
        ("Medium (7 tokens)", "Thank you very much."),
        ("Long (16 tokens)", "The scientific method is a systematic way of learning about the world.")
    ]
    
    print(f"{'Test':<20} {'Baseline':<15} {'Flash':<15} {'Speedup':<10} {'Enc Speedup':<12}")
    print("-" * 80)
    
    for name, text in test_cases:
        baseline_total, baseline_enc, baseline_dec = benchmark_engine("pico_nllb_baseline", text, n_runs=5)
        flash_total, flash_enc, flash_dec = benchmark_engine("pico_nllb_flash", text, n_runs=5)
        
        if baseline_total and flash_total:
            speedup = baseline_total / flash_total
            enc_speedup = baseline_enc / flash_enc if flash_enc > 0 else 1.0
            
            print(f"{name:<20} {baseline_total:>6}ms        {flash_total:>6}ms        {speedup:>6.2f}x    {enc_speedup:>8.2f}x")
            print(f"{'  Encoder:':<20} {baseline_enc:>6}ms        {flash_enc:>6}ms")
            print(f"{'  Decoder:':<20} {baseline_dec:>6}ms        {flash_dec:>6}ms")
            print()
        else:
            print(f"{name:<20} ERROR")
    
    print("=" * 80)
    print("\nExpected from Roadmap: 5-10% speedup (1.05-1.10x)")
    print("Flash Attention reduces memory bandwidth by fusing softmax + value accumulation")
    print("=" * 80)

if __name__ == "__main__":
    main()
