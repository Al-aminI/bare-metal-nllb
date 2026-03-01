/*  loader.c -- PicoNLLB
 *
 *  Opens .safetensors with mmap(), parses the JSON header to find byte
 *  offsets, and wires every tensor pointer directly into the mapped region.
 *  Zero heap allocation for weights.
 *
 *  Safetensors layout:
 *    [8 bytes]           uint64_t header_size (little-endian)
 *    [header_size bytes] JSON string
 *    [rest]              raw tensor data ("data region")
 */

#include "pico.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ─── Minimal JSON helpers ──────────────────────────────────────────────── */

static const char* find_key(const char* json, size_t json_len, const char* key) {
    size_t klen = strlen(key);
    const char* p   = json;
    const char* end = json + json_len;
    while (p < end - klen - 2) {
        if (*p == '"' && memcmp(p + 1, key, klen) == 0 && p[klen + 1] == '"') {
            const char* q = p + klen + 2;
            while (q < end && *q != '{') q++;
            if (q < end) return q;
        }
        p++;
    }
    return NULL;
}

static int parse_data_offsets(const char* obj, size_t max_scan,
                              uint64_t* off_start, uint64_t* off_end) {
    const char* p   = obj;
    const char* end = obj + max_scan;
    while (p < end - 13) {
        if (memcmp(p, "data_offsets", 12) == 0) {
            while (p < end && *p != '[') p++;
            if (p >= end) return -1;
            p++;
            while (p < end && (*p == ' ' || *p == '\n')) p++;
            *off_start = (uint64_t)strtoull(p, NULL, 10);
            while (p < end && *p != ',') p++;
            p++;
            while (p < end && (*p == ' ' || *p == '\n')) p++;
            *off_end = (uint64_t)strtoull(p, NULL, 10);
            return 0;
        }
        p++;
    }
    return -1;
}

static const void* resolve(const char* json, size_t json_len,
                            const char* key,
                            const uint8_t* data_region,
                            size_t* out_nbytes) {
    const char* obj = find_key(json, json_len, key);
    if (!obj) {
        if (out_nbytes) *out_nbytes = 0;
        return NULL;
    }
    uint64_t s = 0, e = 0;
    if (parse_data_offsets(obj, 512, &s, &e) != 0) {
        fprintf(stderr, "[loader] ERROR: bad offsets for: %s\n", key);
        return NULL;
    }
    if (out_nbytes) *out_nbytes = (size_t)(e - s);
    return (const void*)(data_region + s);
}

#define RESOLVE(key)        resolve(json, json_len, key, data_region, NULL)
#define RESOLVE_SZ(key, sz) resolve(json, json_len, key, data_region, sz)

/* ─── Wire INT8 linear layers from CTranslate2 layout ───────────────────── */

static void wire_linear_int8(LinearInt8* lin,
                             const char* json, size_t json_len,
                             const uint8_t* data_region,
                             const char* prefix,
                             int out_features,
                             int in_features) {
    char key[256];

    snprintf(key, sizeof(key), "%s/weight", prefix);
    lin->weight = (const int8_t*)RESOLVE(key);

    snprintf(key, sizeof(key), "%s/weight_scale", prefix);
    lin->scale = (const float*)RESOLVE(key);

    snprintf(key, sizeof(key), "%s/bias", prefix);
    lin->bias = (const float*)RESOLVE(key);

    lin->out_features = out_features;
    lin->in_features  = in_features;
}

/* ─── Wire encoder layer from CT2 spec ───────────────────────────────────── */

static void wire_encoder_layer(TransformerLayer* tl,
                               const char* json, size_t json_len,
                               const uint8_t* data_region,
                               int layer_idx) {
    char key[256];

    /* Self-attention: linear_0 is fused QKV: [3*D_MODEL, D_MODEL]. */
    snprintf(key, sizeof(key), "encoder/layer_%d/self_attention/linear_0", layer_idx);
    char wkey[256], skey[256], bkey[256];
    snprintf(wkey, sizeof(wkey), "%s/weight", key);
    snprintf(skey, sizeof(skey), "%s/weight_scale", key);
    snprintf(bkey, sizeof(bkey), "%s/bias", key);
    const int8_t* fused_w = (const int8_t*)RESOLVE(wkey);
    const float*  fused_s = (const float*)RESOLVE(skey);
    const float*  fused_b = (const float*)RESOLVE(bkey);

    if (!fused_w || !fused_s || !fused_b) {
        fprintf(stderr, "[loader] FATAL: encoder self_attention linear_0 missing for layer %d\n",
                layer_idx);
        exit(1);
    }

    /* Q, K, V views into fused rows. */
    tl->self_attn.q_proj.weight = fused_w;
    tl->self_attn.q_proj.scale  = fused_s;
    tl->self_attn.q_proj.bias   = fused_b;
    tl->self_attn.q_proj.out_features = D_MODEL;
    tl->self_attn.q_proj.in_features  = D_MODEL;

    tl->self_attn.k_proj.weight = fused_w + (size_t)D_MODEL * D_MODEL;
    tl->self_attn.k_proj.scale  = fused_s + D_MODEL;
    tl->self_attn.k_proj.bias   = fused_b + D_MODEL;
    tl->self_attn.k_proj.out_features = D_MODEL;
    tl->self_attn.k_proj.in_features  = D_MODEL;

    tl->self_attn.v_proj.weight = fused_w + (size_t)2 * D_MODEL * D_MODEL;
    tl->self_attn.v_proj.scale  = fused_s + 2 * D_MODEL;
    tl->self_attn.v_proj.bias   = fused_b + 2 * D_MODEL;
    tl->self_attn.v_proj.out_features = D_MODEL;
    tl->self_attn.v_proj.in_features  = D_MODEL;

    /* Self-attention out projection: linear_1 [D_MODEL, D_MODEL]. */
    snprintf(key, sizeof(key), "encoder/layer_%d/self_attention/linear_1", layer_idx);
    wire_linear_int8(&tl->self_attn.out_proj, json, json_len, data_region,
                     key, D_MODEL, D_MODEL);

    /* No encoder-attn in encoder stack: zero-init. */
    memset(&tl->encoder_attn, 0, sizeof(tl->encoder_attn));

    /* FFN: linear_0 [FFN_DIM, D_MODEL], linear_1 [D_MODEL, FFN_DIM]. */
    snprintf(key, sizeof(key), "encoder/layer_%d/ffn/linear_0", layer_idx);
    wire_linear_int8(&tl->fc1, json, json_len, data_region,
                     key, FFN_DIM, D_MODEL);
    snprintf(key, sizeof(key), "encoder/layer_%d/ffn/linear_1", layer_idx);
    wire_linear_int8(&tl->fc2, json, json_len, data_region,
                     key, D_MODEL, FFN_DIM);

    /* Layer norms. */
    snprintf(key, sizeof(key), "encoder/layer_%d/self_attention/layer_norm/gamma", layer_idx);
    tl->self_attn_layer_norm_gamma = (const float*)RESOLVE(key);
    snprintf(key, sizeof(key), "encoder/layer_%d/self_attention/layer_norm/beta", layer_idx);
    tl->self_attn_layer_norm_beta = (const float*)RESOLVE(key);

    snprintf(key, sizeof(key), "encoder/layer_%d/ffn/layer_norm/gamma", layer_idx);
    tl->final_layer_norm_gamma = (const float*)RESOLVE(key);
    snprintf(key, sizeof(key), "encoder/layer_%d/ffn/layer_norm/beta", layer_idx);
    tl->final_layer_norm_beta = (const float*)RESOLVE(key);

    tl->encoder_attn_layer_norm_gamma = NULL;
    tl->encoder_attn_layer_norm_beta  = NULL;
}

/* ─── Wire decoder layer from CT2 spec ───────────────────────────────────── */

static void wire_decoder_layer(TransformerLayer* tl,
                               const char* json, size_t json_len,
                               const uint8_t* data_region,
                               int layer_idx) {
    char key[256];

    /* Self-attention: fused QKV linear_0 and out linear_1. */
    snprintf(key, sizeof(key), "decoder/layer_%d/self_attention/linear_0", layer_idx);
    char wkey[256], skey[256], bkey[256];
    snprintf(wkey, sizeof(wkey), "%s/weight", key);
    snprintf(skey, sizeof(skey), "%s/weight_scale", key);
    snprintf(bkey, sizeof(bkey), "%s/bias", key);
    const int8_t* fused_w = (const int8_t*)RESOLVE(wkey);
    const float*  fused_s = (const float*)RESOLVE(skey);
    const float*  fused_b = (const float*)RESOLVE(bkey);

    if (!fused_w || !fused_s || !fused_b) {
        fprintf(stderr, "[loader] FATAL: decoder self_attention linear_0 missing for layer %d\n",
                layer_idx);
        exit(1);
    }

    tl->self_attn.q_proj.weight = fused_w;
    tl->self_attn.q_proj.scale  = fused_s;
    tl->self_attn.q_proj.bias   = fused_b;
    tl->self_attn.q_proj.out_features = D_MODEL;
    tl->self_attn.q_proj.in_features  = D_MODEL;

    tl->self_attn.k_proj.weight = fused_w + (size_t)D_MODEL * D_MODEL;
    tl->self_attn.k_proj.scale  = fused_s + D_MODEL;
    tl->self_attn.k_proj.bias   = fused_b + D_MODEL;
    tl->self_attn.k_proj.out_features = D_MODEL;
    tl->self_attn.k_proj.in_features  = D_MODEL;

    tl->self_attn.v_proj.weight = fused_w + (size_t)2 * D_MODEL * D_MODEL;
    tl->self_attn.v_proj.scale  = fused_s + 2 * D_MODEL;
    tl->self_attn.v_proj.bias   = fused_b + 2 * D_MODEL;
    tl->self_attn.v_proj.out_features = D_MODEL;
    tl->self_attn.v_proj.in_features  = D_MODEL;

    snprintf(key, sizeof(key), "decoder/layer_%d/self_attention/linear_1", layer_idx);
    wire_linear_int8(&tl->self_attn.out_proj, json, json_len, data_region,
                     key, D_MODEL, D_MODEL);

    /* Encoder attention:
     *   attention/linear_0: Q   [D_MODEL, D_MODEL]
     *   attention/linear_1: KV  [2*D_MODEL, D_MODEL] (first K, then V)
     *   attention/linear_2: out [D_MODEL, D_MODEL]
     */
    snprintf(key, sizeof(key), "decoder/layer_%d/attention/linear_0", layer_idx);
    wire_linear_int8(&tl->encoder_attn.q_proj, json, json_len, data_region,
                     key, D_MODEL, D_MODEL);

    snprintf(key, sizeof(key), "decoder/layer_%d/attention/linear_1", layer_idx);
    char kv_wkey[256], kv_skey[256], kv_bkey[256];
    snprintf(kv_wkey, sizeof(kv_wkey), "%s/weight", key);
    snprintf(kv_skey, sizeof(kv_skey), "%s/weight_scale", key);
    snprintf(kv_bkey, sizeof(kv_bkey), "%s/bias", key);
    const int8_t* kv_w = (const int8_t*)RESOLVE(kv_wkey);
    const float*  kv_s = (const float*)RESOLVE(kv_skey);
    const float*  kv_b = (const float*)RESOLVE(kv_bkey);
    if (!kv_w || !kv_s || !kv_b) {
        fprintf(stderr, "[loader] FATAL: decoder attention linear_1 missing for layer %d\n",
                layer_idx);
        exit(1);
    }
    tl->encoder_attn.k_proj.weight = kv_w;
    tl->encoder_attn.k_proj.scale  = kv_s;
    tl->encoder_attn.k_proj.bias   = kv_b;
    tl->encoder_attn.k_proj.out_features = D_MODEL;
    tl->encoder_attn.k_proj.in_features  = D_MODEL;

    tl->encoder_attn.v_proj.weight = kv_w + (size_t)D_MODEL * D_MODEL;
    tl->encoder_attn.v_proj.scale  = kv_s + D_MODEL;
    tl->encoder_attn.v_proj.bias   = kv_b + D_MODEL;
    tl->encoder_attn.v_proj.out_features = D_MODEL;
    tl->encoder_attn.v_proj.in_features  = D_MODEL;

    snprintf(key, sizeof(key), "decoder/layer_%d/attention/linear_2", layer_idx);
    wire_linear_int8(&tl->encoder_attn.out_proj, json, json_len, data_region,
                     key, D_MODEL, D_MODEL);

    /* FFN. */
    snprintf(key, sizeof(key), "decoder/layer_%d/ffn/linear_0", layer_idx);
    wire_linear_int8(&tl->fc1, json, json_len, data_region,
                     key, FFN_DIM, D_MODEL);
    snprintf(key, sizeof(key), "decoder/layer_%d/ffn/linear_1", layer_idx);
    wire_linear_int8(&tl->fc2, json, json_len, data_region,
                     key, D_MODEL, FFN_DIM);

    /* Layer norms. */
    snprintf(key, sizeof(key), "decoder/layer_%d/self_attention/layer_norm/gamma", layer_idx);
    tl->self_attn_layer_norm_gamma = (const float*)RESOLVE(key);
    snprintf(key, sizeof(key), "decoder/layer_%d/self_attention/layer_norm/beta", layer_idx);
    tl->self_attn_layer_norm_beta = (const float*)RESOLVE(key);

    snprintf(key, sizeof(key), "decoder/layer_%d/attention/layer_norm/gamma", layer_idx);
    tl->encoder_attn_layer_norm_gamma = (const float*)RESOLVE(key);
    snprintf(key, sizeof(key), "decoder/layer_%d/attention/layer_norm/beta", layer_idx);
    tl->encoder_attn_layer_norm_beta = (const float*)RESOLVE(key);

    snprintf(key, sizeof(key), "decoder/layer_%d/ffn/layer_norm/gamma", layer_idx);
    tl->final_layer_norm_gamma = (const float*)RESOLVE(key);
    snprintf(key, sizeof(key), "decoder/layer_%d/ffn/layer_norm/beta", layer_idx);
    tl->final_layer_norm_beta = (const float*)RESOLVE(key);
}

/* ─── Public API ─────────────────────────────────────────────────────────── */

int pico_load(PicoModel* m, const char* path) {
    memset(m, 0, sizeof(PicoModel));

    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) { perror("[loader] open"); return -1; }

    struct stat st;
    fstat(m->fd, &st);
    m->mmap_size = (size_t)st.st_size;

    m->mmap_ptr = mmap(NULL, m->mmap_size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->mmap_ptr == MAP_FAILED) {
        perror("[loader] mmap");
        close(m->fd);
        return -1;
    }
    madvise(m->mmap_ptr, m->mmap_size, MADV_RANDOM);

    printf("[loader] mmap'd %zu MB from %s\n", m->mmap_size / (1024*1024), path);

    const uint8_t* base = (const uint8_t*)m->mmap_ptr;
    uint64_t header_size = 0;
    memcpy(&header_size, base, 8);

    const char*    json        = (const char*)(base + 8);
    size_t         json_len    = (size_t)header_size;
    const uint8_t* data_region = base + 8 + header_size;

    printf("[loader] header: %llu bytes, data at offset %llu\n",
           (unsigned long long)header_size,
           (unsigned long long)(8 + header_size));

    /* Embeddings (INT8 + per-row scale).
     * NLLB shares embeddings between encoder and decoder.
     * Both use the same int8 weights and per-row scales. */
    m->encoder_embed_weight =
        (const int8_t*)RESOLVE("encoder/embeddings_0/weight");
    m->decoder_embed_weight =
        (const int8_t*)RESOLVE("decoder/embeddings/weight");
    m->decoder_embed_scale =
        (const float*)RESOLVE("decoder/embeddings/weight_scale");
    
    /* Encoder uses the same per-row scales as decoder (shared embeddings) */
    m->encoder_embed_scale_array = m->decoder_embed_scale;
    
    /* The encoder/scale_embeddings is the sqrt(d_model) multiplier, not a quantization scale */
    const float* enc_scale_tensor =
        (const float*)RESOLVE("encoder/scale_embeddings");
    m->encoder_embed_scale_multiplier = enc_scale_tensor ? enc_scale_tensor[0] : sqrtf((float)D_MODEL);

    printf("[loader] enc_emb_w=%p enc_scale_array=%p enc_scale_mult=%f dec_emb_w=%p dec_scale=%p\n",
           (const void*)m->encoder_embed_weight,
           (const void*)m->encoder_embed_scale_array,
           m->encoder_embed_scale_multiplier,
           (const void*)m->decoder_embed_weight,
           (const void*)m->decoder_embed_scale);

    if (!m->encoder_embed_weight || !m->encoder_embed_scale_array
        || !m->decoder_embed_weight || !m->decoder_embed_scale) {
        fprintf(stderr, "[loader] FATAL: int8 embeddings not found\n");
        return -1;
    }

    /* Encoder layers. */
    for (int i = 0; i < ENCODER_LAYERS; i++)
        wire_encoder_layer(&m->encoder_layers[i], json, json_len, data_region, i);
    printf("[loader] wired %d encoder layers\n", ENCODER_LAYERS);

    m->encoder_layernorm_gamma =
        (const float*)RESOLVE("encoder/layer_norm/gamma");
    m->encoder_layernorm_beta =
        (const float*)RESOLVE("encoder/layer_norm/beta");

    /* Decoder layers. */
    for (int i = 0; i < DECODER_LAYERS; i++)
        wire_decoder_layer(&m->decoder_layers[i], json, json_len, data_region, i);
    printf("[loader] wired %d decoder layers\n", DECODER_LAYERS);

    m->decoder_layernorm_gamma =
        (const float*)RESOLVE("decoder/layer_norm/gamma");
    m->decoder_layernorm_beta =
        (const float*)RESOLVE("decoder/layer_norm/beta");

    printf("[loader] model fully wired (INT8)\n");
    return 0;
}

void pico_free(PicoModel* m) {
    if (m->mmap_ptr && m->mmap_ptr != MAP_FAILED)
        munmap(m->mmap_ptr, m->mmap_size);
    if (m->fd > 0)
        close(m->fd);
    memset(m, 0, sizeof(PicoModel));
}
