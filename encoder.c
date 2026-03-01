/*  encoder.c -- PicoNLLB
 *
 *  NLLB encoder: embed -> positional -> 12x TransformerLayer -> final_layernorm
 *
 *  Each TransformerLayer:
 *    residual = x
 *    x = LayerNorm(x)
 *    x = residual + MultiHeadSelfAttention(x)
 *    residual = x
 *    x = LayerNorm(x)
 *    x = residual + FFN(x)
 *
 *  All large buffers are heap-allocated once in pico_encode() and threaded
 *  through layer calls to avoid stack overflow.
 */

#include "pico.h"

/* ─── NLLB/M2M100 sinusoidal positional encoding ────────────────────────── 
 *
 * NLLB uses M2M100SinusoidalPositionalEmbedding which:
 *   - Offsets positions by (padding_idx + 1) = 2
 *   - Uses freq = exp(-i * log(10000) / (half_dim - 1))
 *   - Layout: first half dims = sin, second half = cos
 */

static void add_sinusoidal_pos(float* x, int pos, int d_model) {
    int half = d_model / 2;
    int actual_pos = pos + 2;  /* NLLB padding_idx=1, offset=padding_idx+1=2 */
    for (int i = 0; i < half; i++) {
        float freq = expf(-(float)i * logf(10000.0f) / (float)(half - 1));
        float angle = (float)actual_pos * freq;
        x[i]        += sinf(angle);
        x[i + half] += cosf(angle);
    }
}

static void add_positional_encoding(float* x, int pos, int d_model,
                                     const uint16_t* learned_pos) {
    (void)learned_pos;
    add_sinusoidal_pos(x, pos, d_model);
}

/* ─── Encoder scratch buffer layout ─────────────────────────────────────── 
 *
 * We heap-allocate one big buffer and carve it into regions:
 *   residual  [seq_len * D_MODEL]
 *   normed    [seq_len * D_MODEL]
 *   attn_out  [seq_len * D_MODEL]
 *   Q         [seq_len * D_MODEL]
 *   K         [seq_len * D_MODEL]
 *   V         [seq_len * D_MODEL]
 *   head_out  [seq_len * D_MODEL]
 *   scores    [seq_len]             (flash: only one row at a time)
 *   ffn_buf   [FFN_DIM]
 *   tmp       [FFN_DIM]              (dequant scratch)
 */

typedef struct {
    float* residual;   /* [seq * D] */
    float* normed;     /* [seq * D] */
    float* attn_out;   /* [seq * D] */
    float* Q;          /* [seq * D] */
    float* K;          /* [seq * D] */
    float* V;          /* [seq * D] */
    float* head_out;   /* [seq * D] */
    float* scores;     /* [seq] - only one row for flash attention */
    float* ffn_buf;    /* [FFN_DIM] */
    float* tmp;        /* [FFN_DIM] */
} EncoderScratch;

static EncoderScratch alloc_encoder_scratch(int seq_len) {
    size_t sd = (size_t)seq_len * D_MODEL;
    size_t ss = (size_t)seq_len;  /* Flash: only one row */
    size_t total = 7 * sd + ss + 2 * FFN_DIM;

    float* buf = (float*)malloc(total * sizeof(float));
    if (!buf) {
        fprintf(stderr, "[encoder] OOM scratch (%zu floats)\n", total);
        exit(1);
    }

    EncoderScratch s;
    float* p = buf;
    s.residual = p; p += sd;
    s.normed   = p; p += sd;
    s.attn_out = p; p += sd;
    s.Q        = p; p += sd;
    s.K        = p; p += sd;
    s.V        = p; p += sd;
    s.head_out = p; p += sd;
    s.scores   = p; p += ss;
    s.ffn_buf  = p; p += FFN_DIM;
    s.tmp      = p;
    return s;
}

static void free_encoder_scratch(EncoderScratch* s) {
    free(s->residual);
    memset(s, 0, sizeof(*s));
}

/* ─── Multi-head self-attention ─────────────────────────────────────────── */

static void multi_head_self_attn(const Attention* attn,
                                  int              seq_len,
                                  const float*     hidden,
                                  float*           out,
                                  EncoderScratch*  sc) {
    int dm = D_MODEL;
    int hd = HEAD_DIM;

    for (int t = 0; t < seq_len; t++) {
        mat_vec_int8(&attn->q_proj, hidden + t * dm, sc->Q + t * dm);
        mat_vec_int8(&attn->k_proj, hidden + t * dm, sc->K + t * dm);
        mat_vec_int8(&attn->v_proj, hidden + t * dm, sc->V + t * dm);
    }

    for (int h = 0; h < N_HEADS; h++) {
        float scale = 1.0f / sqrtf((float)hd);

        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                const float* qi = sc->Q + i * dm + h * hd;
                const float* kj = sc->K + j * dm + h * hd;
                for (int d = 0; d < hd; d++) dot += qi[d] * kj[d];
                sc->scores[i * seq_len + j] = dot * scale;
            }
            softmax(sc->scores + i * seq_len, seq_len);
        }

        for (int i = 0; i < seq_len; i++) {
            float* oi = sc->head_out + i * dm + h * hd;
            memset(oi, 0, hd * sizeof(float));
            for (int j = 0; j < seq_len; j++) {
                float a = sc->scores[i * seq_len + j];
                const float* vj = sc->V + j * dm + h * hd;
                for (int d = 0; d < hd; d++) oi[d] += a * vj[d];
            }
        }
    }

    for (int t = 0; t < seq_len; t++)
        mat_vec_int8(&attn->out_proj, sc->head_out + t * dm, out + t * dm);
}

/* ─── Multi-head self-attention with Flash Attention ────────────────────── */

static void multi_head_self_attn_flash(const Attention* attn,
                                        int              seq_len,
                                        const float*     hidden,
                                        float*           out,
                                        EncoderScratch*  sc) {
    int dm = D_MODEL;
    int hd = HEAD_DIM;

    /* Project Q, K, V */
    for (int t = 0; t < seq_len; t++) {
        mat_vec_int8(&attn->q_proj, hidden + t * dm, sc->Q + t * dm);
        mat_vec_int8(&attn->k_proj, hidden + t * dm, sc->K + t * dm);
        mat_vec_int8(&attn->v_proj, hidden + t * dm, sc->V + t * dm);
    }

    /* Flash attention: fused softmax + value accumulation per head */
    for (int h = 0; h < N_HEADS; h++) {
        float scale = 1.0f / sqrtf((float)hd);

        /* Process each query independently */
        for (int i = 0; i < seq_len; i++) {
            const float* qi = sc->Q + i * dm + h * hd;
            float* oi = sc->head_out + i * dm + h * hd;
            
            /* Find max score for numerical stability */
            float max_score = -1e30f;
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                const float* kj = sc->K + j * dm + h * hd;
                for (int d = 0; d < hd; d++) {
                    dot += qi[d] * kj[d];
                }
                sc->scores[j] = dot * scale;
                if (sc->scores[j] > max_score) max_score = sc->scores[j];
            }
            
            /* Fused: compute exp, accumulate values, track sum */
            float sum_exp = 0.0f;
            memset(oi, 0, hd * sizeof(float));
            for (int j = 0; j < seq_len; j++) {
                float exp_score = expf(sc->scores[j] - max_score);
                sum_exp += exp_score;
                
                const float* vj = sc->V + j * dm + h * hd;
                for (int d = 0; d < hd; d++) {
                    oi[d] += exp_score * vj[d];
                }
            }
            
            /* Normalize */
            float inv_sum = 1.0f / sum_exp;
            for (int d = 0; d < hd; d++) {
                oi[d] *= inv_sum;
            }
        }
    }

    /* Output projection */
    for (int t = 0; t < seq_len; t++)
        mat_vec_int8(&attn->out_proj, sc->head_out + t * dm, out + t * dm);
}

/* ─── Single encoder layer ──────────────────────────────────────────────── */

static void encoder_layer_forward(const TransformerLayer* layer,
                                   int                     seq_len,
                                   float*                  x,
                                   EncoderScratch*         sc) {
    int sd = seq_len * D_MODEL;

    /* Sub-layer 1: self-attention with flash attention */
    memcpy(sc->residual, x, sd * sizeof(float));
        for (int t = 0; t < seq_len; t++)
        layernorm(sc->normed + t * D_MODEL, x + t * D_MODEL,
                  layer->self_attn_layer_norm_gamma,
                  layer->self_attn_layer_norm_beta, D_MODEL);

    multi_head_self_attn_flash(&layer->self_attn, seq_len, sc->normed, sc->attn_out, sc);

    residual_add(sc->attn_out, sc->residual, sd);
    memcpy(x, sc->attn_out, sd * sizeof(float));

    /* Sub-layer 2: FFN */
    memcpy(sc->residual, x, sd * sizeof(float));
    for (int t = 0; t < seq_len; t++) {
        layernorm(sc->normed + t * D_MODEL, x + t * D_MODEL,
                  layer->final_layer_norm_gamma,
                  layer->final_layer_norm_beta, D_MODEL);

        mat_vec_int8(&layer->fc1, sc->normed + t * D_MODEL, sc->ffn_buf);
        relu(sc->ffn_buf, FFN_DIM);
        mat_vec_int8(&layer->fc2, sc->ffn_buf, sc->attn_out + t * D_MODEL);
    }

    residual_add(sc->attn_out, sc->residual, sd);
    memcpy(x, sc->attn_out, sd * sizeof(float));
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

void pico_encode(PicoModel*  m,
                 const int*  tokens,
                 int         n_tokens,
                 float*      encoder_out) {

    printf("[encoder] encoding %d tokens...\n", n_tokens);

    EncoderScratch sc = alloc_encoder_scratch(n_tokens);

    /* 1. Embedding + scale + positional
     * NLLB order: embed * sqrt(d_model), THEN add positions
     * Encoder uses per-row int8 scales (shared with decoder) */
    float scale = m->encoder_embed_scale_multiplier;  // sqrt(d_model) = 32.0
    for (int t = 0; t < n_tokens; t++) {
        encoder_embed_lookup(m->encoder_embed_weight, m->encoder_embed_scale_array,
                             tokens[t], encoder_out + t * D_MODEL);
        for (int i = 0; i < D_MODEL; i++)
            encoder_out[t * D_MODEL + i] *= scale;
        add_positional_encoding(encoder_out + t * D_MODEL, t, D_MODEL,
                                 NULL);
    }

    /* 3. Encoder layers */
    for (int l = 0; l < ENCODER_LAYERS; l++) {
        printf("[encoder] layer %d/%d\n", l + 1, ENCODER_LAYERS);
        encoder_layer_forward(&m->encoder_layers[l], n_tokens, encoder_out, &sc);
    }

    /* 4. Final layernorm */
    for (int t = 0; t < n_tokens; t++)
        layernorm(sc.normed + t * D_MODEL, encoder_out + t * D_MODEL,
                  m->encoder_layernorm_gamma, m->encoder_layernorm_beta,
                  D_MODEL);
    memcpy(encoder_out, sc.normed, n_tokens * D_MODEL * sizeof(float));

    free_encoder_scratch(&sc);
    printf("[encoder] done -- output [%d, %d]\n", n_tokens, D_MODEL);
}
