/*  decoder.c -- MetalNLLB
 *
 *  Decoder forward pass and vocab projection.
 *  Beam search logic lives in main.c; this file provides the building blocks.
 *
 *  KV caches:
 *    - Self-attention: kv_cache[layer][K|V][head][token][dim]
 *      Per-beam, caller-managed.
 *    - Cross-attention: xattn_kv_cache[layer][K|V][src_token][D_MODEL]
 *      Shared across beams, projected once on step 0.
 */

#include "pico.h"
#include <pthread.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

/* ─── Self-attention KV cache layout ────────────────────────────────────── */

#define SA_K_OFFSET(l)  ((l) * 2 * N_HEADS * MAX_GEN_LEN * HEAD_DIM)
#define SA_V_OFFSET(l)  (SA_K_OFFSET(l) + N_HEADS * MAX_GEN_LEN * HEAD_DIM)
#define SA_HEAD_STRIDE  (MAX_GEN_LEN * HEAD_DIM)

/* ─── Cross-attention KV cache layout ───────────────────────────────────── */

#define XA_LAYER_STRIDE  (2 * MAX_SEQ_LEN * D_MODEL)
#define XA_K_OFFSET(l)   ((l) * XA_LAYER_STRIDE)
#define XA_V_OFFSET(l)   (XA_K_OFFSET(l) + MAX_SEQ_LEN * D_MODEL)

/* ─── Cache size queries ────────────────────────────────────────────────── */

size_t pico_kv_cache_size(void) {
    return KV_CACHE_TOTAL_FLOATS * sizeof(float);
}

size_t pico_xattn_cache_size(void) {
    return XATTN_CACHE_TOTAL_FLOATS * sizeof(float);
}

/* ─── Masked self-attention with KV cache ───────────────────────────────── */

static void decoder_self_attn_cached(const Attention* attn,
                                      const float*     x,
                                      float*           out,
                                      float*           kv_cache,
                                      int              layer_idx,
                                      int              step,
                                      float*           tmp) {
    float q_full[D_MODEL], k_full[D_MODEL], v_full[D_MODEL];

    mat_vec_int8(&attn->q_proj, x, q_full);
    mat_vec_int8(&attn->k_proj, x, k_full);
    mat_vec_int8(&attn->v_proj, x, v_full);

    float* K_cache = kv_cache + SA_K_OFFSET(layer_idx);
    float* V_cache = kv_cache + SA_V_OFFSET(layer_idx);

    for (int h = 0; h < N_HEADS; h++) {
        memcpy(K_cache + h * SA_HEAD_STRIDE + step * HEAD_DIM,
               k_full + h * HEAD_DIM, HEAD_DIM * sizeof(float));
        memcpy(V_cache + h * SA_HEAD_STRIDE + step * HEAD_DIM,
               v_full + h * HEAD_DIM, HEAD_DIM * sizeof(float));
    }

    int ctx_len = step + 1;
    float attn_out_full[D_MODEL];

    for (int h = 0; h < N_HEADS; h++) {
        const float* qh = q_full + h * HEAD_DIM;
        const float* Kh = K_cache + h * SA_HEAD_STRIDE;
        const float* Vh = V_cache + h * SA_HEAD_STRIDE;
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        float scores[MAX_GEN_LEN];

        for (int j = 0; j < ctx_len; j++) {
            float dot = 0.0f;
            const float* kj = Kh + j * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * kj[d];
            scores[j] = dot * scale;
        }
        softmax(scores, ctx_len);

        float out_h[HEAD_DIM];
        memset(out_h, 0, sizeof(out_h));
        for (int j = 0; j < ctx_len; j++) {
            const float* vj = Vh + j * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) out_h[d] += scores[j] * vj[d];
        }
        memcpy(attn_out_full + h * HEAD_DIM, out_h, HEAD_DIM * sizeof(float));
    }

    mat_vec_int8(&attn->out_proj, attn_out_full, out);
}

/* ─── Cross-attention with cached encoder K/V ───────────────────────────── */

static void decoder_cross_attn_cached(const Attention* attn,
                                       const float*     x,
                                       const float*     encoder_out,
                                       int              n_src,
                                       float*           out,
                                       float*           xattn_kv_cache,
                                       int              layer_idx,
                                       int              step,
                                       float*           tmp) {
    float q_full[D_MODEL];
    mat_vec_int8(&attn->q_proj, x, q_full);

    float* K = xattn_kv_cache + XA_K_OFFSET(layer_idx);
    float* V = xattn_kv_cache + XA_V_OFFSET(layer_idx);

    if (step == 0) {
        for (int s = 0; s < n_src; s++) {
            mat_vec_int8(&attn->k_proj, encoder_out + s * D_MODEL,
                         K + s * D_MODEL);
            mat_vec_int8(&attn->v_proj, encoder_out + s * D_MODEL,
                         V + s * D_MODEL);
        }
    }

    float attn_out_full[D_MODEL];
    for (int h = 0; h < N_HEADS; h++) {
        const float* qh = q_full + h * HEAD_DIM;
        float scale = 1.0f / sqrtf((float)HEAD_DIM);
        float scores[MAX_SEQ_LEN];

        for (int s = 0; s < n_src; s++) {
            float dot = 0.0f;
            const float* ks = K + s * D_MODEL + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) dot += qh[d] * ks[d];
            scores[s] = dot * scale;
        }

        softmax(scores, n_src);

        float out_h[HEAD_DIM];
        memset(out_h, 0, sizeof(out_h));
        for (int s = 0; s < n_src; s++) {
            const float* vs = V + s * D_MODEL + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) out_h[d] += scores[s] * vs[d];
        }
        memcpy(attn_out_full + h * HEAD_DIM, out_h, HEAD_DIM * sizeof(float));
    }

    mat_vec_int8(&attn->out_proj, attn_out_full, out);
}

/* ─── Single decoder layer (one step) ───────────────────────────────────── */

static void decoder_layer_step(const TransformerLayer* layer,
                                float*                  x,
                                const float*            encoder_out,
                                int                     n_src,
                                float*                  kv_cache,
                                float*                  xattn_kv_cache,
                                int                     layer_idx,
                                int                     step) {
    float residual[D_MODEL], normed[D_MODEL], attn_out[D_MODEL];
    float ffn_buf[FFN_DIM], tmp[FFN_DIM];

    memcpy(residual, x, D_MODEL * sizeof(float));
    layernorm(normed, x,
              layer->self_attn_layer_norm_gamma,
              layer->self_attn_layer_norm_beta, D_MODEL);
    decoder_self_attn_cached(&layer->self_attn, normed, attn_out,
                              kv_cache, layer_idx, step, tmp);
    residual_add(attn_out, residual, D_MODEL);
    memcpy(x, attn_out, D_MODEL * sizeof(float));

    memcpy(residual, x, D_MODEL * sizeof(float));
    layernorm(normed, x,
              layer->encoder_attn_layer_norm_gamma,
              layer->encoder_attn_layer_norm_beta, D_MODEL);
    decoder_cross_attn_cached(&layer->encoder_attn, normed,
                               encoder_out, n_src, attn_out,
                               xattn_kv_cache, layer_idx, step, tmp);
    residual_add(attn_out, residual, D_MODEL);
    memcpy(x, attn_out, D_MODEL * sizeof(float));

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

/* ─── Decoder forward pass (no vocab projection) ────────────────────────── */

void pico_decode_forward(PicoModel*   m,
                         const float* encoder_out,
                         int          n_src,
                         int          cur_token,
                         int          step,
                         float*       kv_cache,
                         float*       xattn_kv_cache,
                         float*       out_normed) {
    float x[D_MODEL];
    decoder_embed_lookup(m->decoder_embed_weight, m->decoder_embed_scale, cur_token, x);

    // Apply embedding scaling (standard in transformers)
    float embed_scale = sqrtf((float)D_MODEL);
    for (int i = 0; i < D_MODEL; i++) x[i] *= embed_scale;

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

    for (int l = 0; l < DECODER_LAYERS; l++)
        decoder_layer_step(&m->decoder_layers[l], x,
                           encoder_out, n_src,
                           kv_cache, xattn_kv_cache, l, step);

    layernorm(out_normed, x,
              m->decoder_layernorm_gamma,
              m->decoder_layernorm_beta, D_MODEL);
}

/* ─── Vocab projection to logits buffer ─────────────────────────────────── */

/* Vocab projection worker thread */
typedef struct {
    const int8_t* weight;
    const float*  scale;
    const float*  normed;
    float*        logits;
    int           start_v;
    int           end_v;
    int           mask_lang;
} vocab_proj_task_t;

static void* vocab_proj_worker(void* arg) {
    vocab_proj_task_t* t = (vocab_proj_task_t*)arg;
    
    for (int v = t->start_v; v < t->end_v; v++) {
        if (t->mask_lang && v >= LANG_TOKEN_START && v < LANG_TOKEN_END) {
            t->logits[v] = -1e30f;
            continue;
        }
        
        const int8_t* row = t->weight + (size_t)v * D_MODEL;
        const float inv_s = 1.0f / t->scale[v];
        
        /* Fused dequant+dot */
        float l = 0.0f;
        
#ifdef __ARM_NEON
        /* NEON SIMD path */
        float32x4_t scale_v = vdupq_n_f32(inv_s);
        float32x4_t sum_v = vdupq_n_f32(0.0f);
        
        int d = 0;
        for (; d + 15 < D_MODEL; d += 16) {
            int8x16_t q = vld1q_s8(row + d);
            
            int16x8_t q16_lo = vmovl_s8(vget_low_s8(q));
            int16x8_t q16_hi = vmovl_s8(vget_high_s8(q));
            
            int32x4_t q32_0 = vmovl_s16(vget_low_s16(q16_lo));
            int32x4_t q32_1 = vmovl_s16(vget_high_s16(q16_lo));
            int32x4_t q32_2 = vmovl_s16(vget_low_s16(q16_hi));
            int32x4_t q32_3 = vmovl_s16(vget_high_s16(q16_hi));
            
            float32x4_t qf_0 = vcvtq_f32_s32(q32_0);
            float32x4_t qf_1 = vcvtq_f32_s32(q32_1);
            float32x4_t qf_2 = vcvtq_f32_s32(q32_2);
            float32x4_t qf_3 = vcvtq_f32_s32(q32_3);
            
            qf_0 = vmulq_f32(qf_0, scale_v);
            qf_1 = vmulq_f32(qf_1, scale_v);
            qf_2 = vmulq_f32(qf_2, scale_v);
            qf_3 = vmulq_f32(qf_3, scale_v);
            
            float32x4_t xv_0 = vld1q_f32(t->normed + d);
            float32x4_t xv_1 = vld1q_f32(t->normed + d + 4);
            float32x4_t xv_2 = vld1q_f32(t->normed + d + 8);
            float32x4_t xv_3 = vld1q_f32(t->normed + d + 12);
            
            sum_v = vmlaq_f32(sum_v, qf_0, xv_0);
            sum_v = vmlaq_f32(sum_v, qf_1, xv_1);
            sum_v = vmlaq_f32(sum_v, qf_2, xv_2);
            sum_v = vmlaq_f32(sum_v, qf_3, xv_3);
        }
        
        l = vaddvq_f32(sum_v);
        
        for (; d < D_MODEL; d++) {
            l += ((float)row[d] * inv_s) * t->normed[d];
        }
#else
        /* Scalar path */
        for (int d = 0; d < D_MODEL; d++) {
            l += ((float)row[d] * inv_s) * t->normed[d];
        }
#endif
        
        t->logits[v] = l;
    }
    
    return NULL;
}

void pico_vocab_project(const PicoModel* m,
                        const float*     normed,
                        float*           logits,
                        int              mask_lang) {
    // Vocab projection: logits = embedding_weights @ normed_hidden_state
    // Parallelized across vocabulary for massive speedup
    
    const int n_threads = NUM_THREADS;
    pthread_t threads[NUM_THREADS];
    vocab_proj_task_t tasks[NUM_THREADS];
    int vocab_per_thread = VOCAB_SIZE / n_threads;
    
    for (int t = 0; t < n_threads; t++) {
        tasks[t] = (vocab_proj_task_t){
            .weight = m->decoder_embed_weight,
            .scale = m->decoder_embed_scale,
            .normed = normed,
            .logits = logits,
            .start_v = t * vocab_per_thread,
            .end_v = (t == n_threads - 1) ? VOCAB_SIZE : (t + 1) * vocab_per_thread,
            .mask_lang = mask_lang
        };
        
        if (t > 0) {
            pthread_create(&threads[t], NULL, vocab_proj_worker, &tasks[t]);
        }
    }
    
    /* Main thread does work too */
    vocab_proj_worker(&tasks[0]);
    
    /* Wait for workers */
    for (int t = 1; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}
