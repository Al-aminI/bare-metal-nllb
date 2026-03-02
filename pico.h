#ifndef PICO_H
#define PICO_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VOCAB_SIZE        256206
#define D_MODEL           1024
#define ENCODER_LAYERS    12
#define DECODER_LAYERS    12
#define N_HEADS           16
#define HEAD_DIM          64
#define FFN_DIM           4096
#define MAX_SEQ_LEN       256
#define MAX_GEN_LEN       64
#define KV_CACHE_LAYER_STRIDE   (2 * N_HEADS * MAX_GEN_LEN * HEAD_DIM)
#define KV_CACHE_TOTAL_FLOATS   (DECODER_LAYERS * KV_CACHE_LAYER_STRIDE)
#define XATTN_CACHE_TOTAL_FLOATS (DECODER_LAYERS * 2 * MAX_SEQ_LEN * D_MODEL)

/* ─── INT8 linear layers ─────────────────────────────── */

typedef struct {
    const int8_t* weight;      /* int8 [out_features, in_features] */
    const float*  scale;       /* float32 [out_features]           */
    const float*  bias;        /* float32 [out_features] or NULL   */
    int           out_features;
    int           in_features;
} LinearInt8;

typedef struct {
    LinearInt8 q_proj;
    LinearInt8 k_proj;
    LinearInt8 v_proj;
    LinearInt8 out_proj;
} Attention;

typedef struct {
    Attention    self_attn;
    Attention    encoder_attn;   /* unused for encoder stack */
    LinearInt8   fc1;
    LinearInt8   fc2;
    const float* self_attn_layer_norm_gamma;
    const float* self_attn_layer_norm_beta;
    const float* final_layer_norm_gamma;
    const float* final_layer_norm_beta;
    const float* encoder_attn_layer_norm_gamma; /* decoder only */
    const float* encoder_attn_layer_norm_beta;  /* decoder only */
} TransformerLayer;

typedef struct {
    void*   mmap_ptr;
    size_t  mmap_size;
    int     fd;

    /* Embeddings are shared int8+scale between encoder and decoder. */
    const int8_t* encoder_embed_weight;        /* [VOCAB_SIZE, D_MODEL] int8 */
    const float*  encoder_embed_scale_array;   /* [VOCAB_SIZE] float32 per-row scales (shared with decoder) */
    float         encoder_embed_scale_multiplier; /* sqrt(d_model) = 32.0 */
    const int8_t* decoder_embed_weight;        /* [VOCAB_SIZE, D_MODEL] int8 (same as encoder) */
    const float*  decoder_embed_scale;         /* [VOCAB_SIZE] float32 per-row scales */

    TransformerLayer encoder_layers[ENCODER_LAYERS];
    const float*  encoder_layernorm_gamma;
    const float*  encoder_layernorm_beta;

    TransformerLayer decoder_layers[DECODER_LAYERS];
    const float*  decoder_layernorm_gamma;
    const float*  decoder_layernorm_beta;
} PicoModel;

int    pico_load(PicoModel* m, const char* safetensors_path);
void   pico_free(PicoModel* m);

void   encoder_embed_lookup(const int8_t* weight, const float* scale_array, int token_id, float* out);
void   decoder_embed_lookup(const int8_t* weight, const float* scale, int token_id, float* out);
void   mat_vec_int8(const LinearInt8* layer, const float* x, float* out);
void   layernorm(float* out, const float* x, const float* gamma, const float* beta, int n);
void   softmax(float* x, int n);
void   relu(float* x, int n);
void   residual_add(float* x, const float* residual, int n);
void   attention_head(const float* Q, const float* K, const float* V,
                      const float* mask, float* out, float* scores_buf,
                      int seq_len, int head_dim);

/* Flash attention: fused computation with online softmax */
void   attention_head_flash(const float* Q, const float* K, const float* V,
                            const float* mask, float* out,
                            int seq_len, int head_dim);

void   pico_encode(PicoModel* m, const int* tokens, int n_tokens, float* encoder_out);

/* Beam search */
#define BEAM_SIZE             4
#define BEAM_TOPK             (BEAM_SIZE * 2)
#define LENGTH_PENALTY        0.0f    /* default: 0 (no normalization) */
#define REPETITION_PENALTY    1.2f
#define NO_REPEAT_NGRAM_SIZE  2       /* block repeated bigrams */
#define LANG_TOKEN_START      256000
#define LANG_TOKEN_END        256206

size_t pico_kv_cache_size(void);
size_t pico_xattn_cache_size(void);

/* FP16 cache sizes (50% smaller) */
size_t pico_kv_cache_size_fp16(void);
size_t pico_xattn_cache_size_fp16(void);

void   pico_decode_forward(PicoModel* m,
                           const float* encoder_out, int n_src,
                           int cur_token, int step,
                           float* kv_cache, float* xattn_kv_cache,
                           float* out_normed);

/* FP16 version (50% memory reduction) */
void   pico_decode_forward_fp16(PicoModel* m,
                                const float* encoder_out, int n_src,
                                int cur_token, int step,
                                uint16_t* kv_cache_fp16, uint16_t* xattn_kv_cache_fp16,
                                float* out_normed);

void   pico_vocab_project(const PicoModel* m,
                          const float* normed,
                          float* logits,
                          int mask_lang);

#endif
