/* Quick test of INT8 model - just encoder + 1 decoder step */
#include "pico.h"
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s model.safetensors\n", argv[0]);
        return 1;
    }

    PicoModel m;
    if (pico_load(&m, argv[1]) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    printf("Model loaded successfully\n");

    // Test tokens: eng_Latn Hello . EOS
    int src_tokens[] = {256047, 59002, 4, 2};
    int n_src = 4;

    // Allocate encoder output
    float* enc_out = malloc(n_src * D_MODEL * sizeof(float));
    
    // Run encoder
    printf("Running encoder...\n");
    pico_encode(&m, src_tokens, n_src, enc_out);
    printf("Encoder done. First 5 values: %.4f %.4f %.4f %.4f %.4f\n",
           enc_out[0], enc_out[1], enc_out[2], enc_out[3], enc_out[4]);

    // Allocate caches
    float* kv_cache = calloc(KV_CACHE_TOTAL_FLOATS, sizeof(float));
    float* xattn_cache = calloc(XATTN_CACHE_TOTAL_FLOATS, sizeof(float));
    float normed[D_MODEL];

    // Run one decoder step with BOS token
    printf("Running decoder step 0 (BOS=2)...\n");
    pico_decode_forward(&m, enc_out, n_src, 2, 0, kv_cache, xattn_cache, normed);
    printf("Decoder done. Normed first 5: %.4f %.4f %.4f %.4f %.4f\n",
           normed[0], normed[1], normed[2], normed[3], normed[4]);

    // Project to vocab
    printf("Computing logits...\n");
    float* logits = malloc(VOCAB_SIZE * sizeof(float));
    pico_vocab_project(&m, normed, logits, 0);
    
    // Find top 5
    printf("Top 5 logits:\n");
    for (int iter = 0; iter < 5; iter++) {
        int best_idx = 0;
        float best_val = logits[0];
        for (int v = 1; v < VOCAB_SIZE; v++) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best_idx = v;
            }
        }
        printf("  %d: %.4f\n", best_idx, best_val);
        logits[best_idx] = -1e30f;  // Mask for next iteration
    }

    free(enc_out);
    free(kv_cache);
    free(xattn_cache);
    free(logits);
    pico_free(&m);

    return 0;
}
