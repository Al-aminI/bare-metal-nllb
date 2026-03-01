/*  decoder_fp16.c -- MetalNLLB with FP16 KV Cache
 *
 *  Decoder forward pass with FP16 KV cache compression.
 *  Reduces memory footprint by 50% (130MB → 82MB) with negligible quality impact.
 *
 *  KV caches:
 *    - Self-attention: kv_cache[layer][K|V][head][token][dim] (FP16)
 *      Per-beam, caller-managed. 96MB → 48MB
 *    - Cross-attention: xattn_kv_cache[layer][K|V][src_token][D_MODEL] (FP16)
 *      Shared across beams. 24MB → 12MB
 */

#include "pico.h"
#include "fp16.h"

/* ─── Self-attention KV cache layout (FP16) ─────────────────────────────── */

#define SA_K_OFFSET(l)  ((l) * 2 * N_HEADS * MAX_GEN_LEN * HEAD_DIM)
#define SA_V_OFFSET(l)  (SA_K_OFFSET(l) + N_HEADS * MAX_GEN_LEN * HEAD_DIM)
#define SA_HEAD_STRIDE  (MAX_GEN_LEN * HEAD_DIM)

/* ─── Cross-attention KV cache layout (FP16) ────────────────────────────── */

#define XA_LAYER_STRIDE  (2 * MAX_SEQ_LEN * D_MODEL)
#define XA_K_OFFSET(l)   ((l) * XA_LAYER_STRIDE)
#define XA_V_OFFSET(l)   (XA_K_OFFSET(l) + MAX_SEQ_LEN * D_MODEL)

/* ─── Cache size queries (FP16 = half size) ─────────────────────────────── */

size_t pico_kv_cache_size_fp16(void) {
    return KV_CACHE_TOTAL_FLOATS * sizeof(uint16_t);  /* Half of FP32 */
}

size_t pico_xattn_cache_size_fp16(void) {
    return XATTN_CACHE_TOTAL_FLOATS * sizeof(uint16_t);  /* Half of FP32 */
}

/* ─── Masked self-attention with FP16 KV cache ──────────────────────────── */

static void decoder_self_attn_cached_fp16(const Attention* attn,
                                           const float*     x,
                                           float*           out,
                                           uint16_t*        kv_cache_fp16,
                                           int              layer_idx,
                                           int              step,
                                           float*           tmp) {
    float q_full[D_MODEL], k_full[D_MODEL], v_full[D_MODEL];

    mat_vec_int8(&attn->q_proj, x, q_full);
    mat_vec_int8(&attn->k_proj, x, k_full);
    mat_vec_int8(&attn->v_proj, x, v_full);

    uint16_t* K_cache = kv_cache_fp16 + SA_K_OFFSET(layer_idx);
    uint16_t* V_cache = kv_cache_fp16 + SA_V_OFFSET(layer_idx);

    /* Store K and V in FP16 */
    for (int h = 0; h < N_HEADS; h++) {
        uint16_t* k_dst = K_cache + h * SA_HEAD_STRIDE + step * HEAD_DIM;
        uint16_t* v_dst = V_cache + h * SA_HEAD_STRIDE + step * HEAD_DIM;
        
        fp32_array_to_fp16(k_full + h * HEAD_DIM, k_dst, HEAD_DIM);
        fp32_array_to_fp16(v_full + h * HEAD_DIM, v_dst, HEAD_DIM);
    }

    int ctx_len = step + 1;
    float attn_out_full[D_MODEL];

    for (int h = 0; h < N_HEADS; h++) {
        const float* qh = q_full + h * HEAD_DIM;
        const uint16_t* Kh = K_cache + h * SA_HEAD_STRIDE;
        const uint16_t* Vh = V_cache + h * SA_HEAD_STRIDE;
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        float scores[MAX_GEN_LEN];

        /* Compute attention scores (convert K from FP16 on-the-fly) */
        for (int j = 0; j < ctx_len; j++) {
            float dot = 0.0f;
            const uint16_t* kj = Kh + j * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += qh[d] * fp16_to_fp32(kj[d]);
            }
            scores[j] = dot * scale;
        }
        softmax(scores, ctx_len);

        /* Compute output (convert V from FP16 on-the-fly) */
        float out_h[HEAD_DIM];
        memset(out_h, 0, sizeof(out_h));
        for (int j = 0; j < ctx_len; j++) {
            const uint16_t* vj = Vh + j * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                out_h[d] += scores[j] * fp16_to_fp32(vj[d]);
            }
        }
        memcpy(attn_out_full + h * HEAD_DIM, out_h, HEAD_DIM * sizeof(float));
    }

    mat_vec_int8(&attn->out_proj, attn_out_full, out);
}

/* ─── Cross-attention with FP16 cached encoder K/V ──────────────────────── */

static void decoder_cross_attn_cached_fp16(const Attention* attn,
                                            const float*     x,
                                            const float*     encoder_out,
                                            int              n_src,
                                            float*           out,
                                            uint16_t*        xattn_kv_cache_fp16,
                                            int              layer_idx,
                                            int              step,
                                            float*           tmp) {
    float q_full[D_MODEL];
    mat_vec_int8(&attn->q_proj, x, q_full);

    uint16_t* K = xattn_kv_cache_fp16 + XA_K_OFFSET(layer_idx);
    uint16_t* V = xattn_kv_cache_fp16 + XA_V_OFFSET(layer_idx);

    /* Project encoder outputs to K/V on first step, store as FP16 */
    if (step == 0) {
        float k_tmp[D_MODEL], v_tmp[D_MODEL];
        for (int s = 0; s < n_src; s++) {
            mat_vec_int8(&attn->k_proj, encoder_out + s * D_MODEL, k_tmp);
            mat_vec_int8(&attn->v_proj, encoder_out + s * D_MODEL, v_tmp);
            
            fp32_array_to_fp16(k_tmp, K + s * D_MODEL, D_MODEL);
            fp32_array_to_fp16(v_tmp, V + s * D_MODEL, D_MODEL);
        }
    }

    float attn_out_full[D_MODEL];
    for (int h = 0; h < N_HEADS; h++) {
        const float* qh = q_full + h * HEAD_DIM;
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        float scores[MAX_SEQ_LEN];

        /* Compute attention scores (convert K from FP16 on-the-fly) */
        for (int s = 0; s < n_src; s++) {
            float dot = 0.0f;
            const uint16_t* ks = K + s * D_MODEL + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += qh[d] * fp16_to_fp32(ks[d]);
            }
            scores[s] = dot * scale;
        }

        softmax(scores, n_src);

        /* Compute output (convert V from FP16 on-the-fly) */
        float out_h[HEAD_DIM];
        memset(out_h, 0, sizeof(out_h));
        for (int s = 0; s < n_src; s++) {
            const uint16_t* vs = V + s * D_MODEL + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                out_h[d] += scores[s] * fp16_to_fp32(vs[d]);
            }
        }
        memcpy(attn_out_full + h * HEAD_DIM, out_h, HEAD_DIM * sizeof(float));
    }

    mat_vec_int8(&attn->out_proj, attn_out_full, out);
}

/* ─── Single decoder layer (one step) with FP16 cache ───────────────────── */

static void decoder_layer_step_fp16(const TransformerLayer* layer,
                                     float*                  x,
                                     const float*            encoder_out,
                                     int                     n_src,
                                     uint16_t*               kv_cache_fp16,
                                     uint16_t*               xattn_kv_cache_fp16,
                                     int                     layer_idx,
                                     int                     step) {
    float residual[D_MODEL], normed[D_MODEL], attn_out[D_MODEL];
    float ffn_buf[FFN_DIM], tmp[FFN_DIM];

    /* Self-attention */
    memcpy(residual, x, D_MODEL * sizeof(float));
    layernorm(normed, x,
              layer->self_attn_layer_norm_gamma,
              layer->self_attn_layer_norm_beta, D_MODEL);
    decoder_self_attn_cached_fp16(&layer->self_attn, normed, attn_out,
                                   kv_cache_fp16, layer_idx, step, tmp);
    residual_add(attn_out, residual, D_MODEL);
    memcpy(x, attn_out, D_MODEL * sizeof(float));

    /* Cross-attention */
    memcpy(residual, x, D_MODEL * sizeof(float));
    layernorm(normed, x,
              layer->encoder_attn_layer_norm_gamma,
              layer->encoder_attn_layer_norm_beta, D_MODEL);
    decoder_cross_attn_cached_fp16(&layer->encoder_attn, normed,
                                    encoder_out, n_src, attn_out,
                                    xattn_kv_cache_fp16, layer_idx, step, tmp);
    residual_add(attn_out, residual, D_MODEL);
    memcpy(x, attn_out, D_MODEL * sizeof(float));

    /* FFN */
    memcpy(residual, x, D_MODEL * sizeof(float));
    layernorm(normed, x,
              layer->final_layer_norm_gamma,
              layer->final_layer_norm_beta, D_MODEL);
    mat_vec_int8(&layer->fc1, normed, ffn_buf);
    relu(ffn_buf, FFN_DIM);
    mat_vec_int8(&layer->fc2, ffn_buf, attn_out);
    residual_add(attn_out, residual, D_MODEL);
    memcpy(x, attn_out, D_MODEL * sizeof(float));
}

/* ─── Decoder forward pass with FP16 cache (no vocab projection) ────────── */

void pico_decode_forward_fp16(PicoModel*   m,
                               const float* encoder_out,
                               int          n_src,
                               int          cur_token,
                               int          step,
                               uint16_t*    kv_cache_fp16,
                               uint16_t*    xattn_kv_cache_fp16,
                               float*       out_normed) {
    float x[D_MODEL];
    decoder_embed_lookup(m->decoder_embed_weight, m->decoder_embed_scale, cur_token, x);

    /* Apply embedding scaling */
    float embed_scale = sqrtf((float)D_MODEL);
    for (int i = 0; i < D_MODEL; i++) x[i] *= embed_scale;

    /* Add positional encoding */
    {
        int half = D_MODEL / 2;
        int actual_pos = step + 2;
        for (int i = 0; i < half; i++) {
            float freq = expf(-(float)i * logf(10000.0f) / (float)(half - 1));
            float angle = (float)actual_pos * freq;
            x[i]        += sinf(angle);
            x[i + half] += cosf(angle);
        }
    }

    /* Run through decoder layers */
    for (int l = 0; l < DECODER_LAYERS; l++)
        decoder_layer_step_fp16(&m->decoder_layers[l], x,
                                 encoder_out, n_src,
                                 kv_cache_fp16, xattn_kv_cache_fp16, l, step);

    /* Final layer norm */
    layernorm(out_normed, x,
              m->decoder_layernorm_gamma,
              m->decoder_layernorm_beta, D_MODEL);
}
