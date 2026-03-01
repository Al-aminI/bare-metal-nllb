/*  main.c -- PicoNLLB  Beam Search (CTranslate2-style)
 *
 *  Usage:
 *    ./pico_nllb <model.safetensors> <src_lang> <tgt_lang> [token_ids...]
 *
 *  Example (translate "Hello." to French):
 *    ./pico_nllb model.safetensors eng_Latn fra_Latn 59002 4 2
 */

#include "pico.h"
#include <time.h>
#include <pthread.h>

typedef struct { const char* name; int token_id; } LangEntry;
static const LangEntry LANG_TABLE[] = {
    {"eng_Latn", 256047}, {"fra_Latn", 256057}, {"spa_Latn", 256168},
    {"deu_Latn", 256050}, {"zho_Hans", 256158}, {"ara_Arab", 256027},
    {"hin_Deva", 256069}, {"por_Latn", 256148}, {"rus_Cyrl", 256154},
    {"jpn_Jpan", 256090}, {"kor_Hang", 256098}, {"swh_Latn", 256175},
    {"hau_Latn", 256066}, {"yor_Latn", 256219}, {"ibo_Latn", 256073},
    {NULL, 0}
};

static int get_lang_token(const char* code) {
    for (int i = 0; LANG_TABLE[i].name; i++)
        if (strcmp(LANG_TABLE[i].name, code) == 0)
            return LANG_TABLE[i].token_id;
    fprintf(stderr, "[main] Unknown language: %s\n", code);
    return -1;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ─── Beam data ─────────────────────────────────────────────────────────── */

typedef struct {
    int   tokens[MAX_GEN_LEN];
    int   n_tokens;
    float score;
    float eos_score;  /* Score of EOS token (excluded from final score like CT2) */
    int   finished;
} Beam;

typedef struct {
    int   beam_idx;
    int   token;
    float score;
} Candidate;

/* ─── Beam processing task for parallel execution ───────────────────────── */

typedef struct {
    PicoModel* model;
    float* encoder_out;
    int n_src;
    int step;
    float* kv_cache;
    float* xattn_cache;
    float* normed;
    float* logits;
    Beam* beam;
    int* tk;
    float* ts;
} beam_task_t;

/* Forward declaration */
static void* beam_worker(void* arg);

/* ─── Thread pool for beam processing ───────────────────────────────────── */

typedef struct {
    pthread_t thread;
    volatile int active;      /* 1 = has work, 0 = idle */
    volatile int shutdown;    /* 1 = exit thread */
    beam_task_t* task;
} worker_thread_t;

static worker_thread_t workers[BEAM_SIZE - 1];  /* Main thread handles beam 0 */

static void* worker_thread_func(void* arg) {
    worker_thread_t* w = (worker_thread_t*)arg;
    
    while (1) {
        /* Spin-wait for work (low latency) */
        while (!w->active && !w->shutdown) {
            /* Yield CPU to avoid busy-wait */
            sched_yield();
        }
        
        if (w->shutdown) break;
        
        /* Do work */
        if (w->task) {
            beam_worker(w->task);
        }
        
        /* Mark as done */
        w->active = 0;
    }
    
    return NULL;
}

static void init_thread_pool(void) {
    for (int i = 0; i < BEAM_SIZE - 1; i++) {
        workers[i].active = 0;
        workers[i].shutdown = 0;
        workers[i].task = NULL;
        pthread_create(&workers[i].thread, NULL, worker_thread_func, &workers[i]);
    }
}

static void shutdown_thread_pool(void) {
    for (int i = 0; i < BEAM_SIZE - 1; i++) {
        workers[i].shutdown = 1;
        pthread_join(workers[i].thread, NULL);
    }
}

static void dispatch_beam_tasks(beam_task_t* tasks, Beam* beams) {
    /* Dispatch tasks to worker threads */
    int worker_idx = 0;
    for (int b = 1; b < BEAM_SIZE; b++) {
        if (!beams[b].finished) {
            workers[worker_idx].task = &tasks[b];
            workers[worker_idx].active = 1;  /* Atomic write */
            worker_idx++;
        }
    }
    
    /* Main thread processes beam 0 */
    if (!beams[0].finished) {
        beam_worker(&tasks[0]);
    }
    
    /* Wait for workers to finish (spin-wait for low latency) */
    worker_idx = 0;
    for (int b = 1; b < BEAM_SIZE; b++) {
        if (!beams[b].finished) {
            while (workers[worker_idx].active) {
                sched_yield();  /* Yield to avoid busy-wait */
            }
            worker_idx++;
        }
    }
}

/* ─── Log-softmax (numerically stable) ──────────────────────────────────── */

static void log_softmax_inplace(float* x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float se = 0.0f;
    for (int i = 0; i < n; i++) se += expf(x[i] - mx);
    float ln = mx + logf(se);
    for (int i = 0; i < n; i++) x[i] -= ln;
}

/* ─── Top-K extraction ──────────────────────────────────────────────────── */

static void find_topk(const float* v, int n, int k,
                      int* out_tok, float* out_sc) {
    for (int i = 0; i < k; i++) out_sc[i] = -1e30f;
    for (int i = 0; i < n; i++) {
        if (v[i] <= out_sc[k - 1]) continue;
        for (int j = 0; j < k; j++) {
            if (v[i] > out_sc[j]) {
                for (int m = k - 1; m > j; m--) {
                    out_sc[m] = out_sc[m - 1];
                    out_tok[m] = out_tok[m - 1];
                }
                out_sc[j] = v[i];
                out_tok[j] = i;
                break;
            }
        }
    }
}

/* ─── Repetition penalty (CTranslate2 style) ───────────────────────────── */

static void apply_repetition_penalty(float* logits, const int* tokens,
                                     int n_tokens, float penalty) {
    for (int g = 2; g < n_tokens; g++) {
        int t = tokens[g];
        if (t == 2) continue;
        if (logits[t] > 0) logits[t] /= penalty;
        else               logits[t] *= penalty;
    }
}

/* ─── No-repeat-ngram (CTranslate2 decoding_utils.cc) ───────────────────── */

static void apply_no_repeat_ngram(float* logits, const int* tokens,
                                  int n_tokens, int ngram_size) {
    if (n_tokens < ngram_size) return;
    int pfx = ngram_size - 1;
    const int* current = tokens + n_tokens - pfx;
    for (int i = 0; i <= n_tokens - ngram_size; i++) {
        int match = 1;
        for (int j = 0; j < pfx; j++) {
            if (tokens[i + j] != current[j]) { match = 0; break; }
        }
        if (match) {
            int blk = tokens[i + pfx];
            if (blk != 2) logits[blk] = -1e30f;
        }
    }
}

/* ─── Length penalty (CTranslate2: score / pow(len, alpha)) ─────────────── */

static float length_norm(float score, int len) {
    if (LENGTH_PENALTY == 0.0f) return score;
    return score / powf((float)len, LENGTH_PENALTY);
}

/* ─── Beam worker function ──────────────────────────────────────────────── */

static void* beam_worker(void* arg) {
    beam_task_t* t = (beam_task_t*)arg;
    
    if (t->beam->finished) {
        return NULL;
    }

    int cur = t->beam->tokens[t->beam->n_tokens - 1];
    pico_decode_forward(t->model, t->encoder_out, t->n_src, cur, t->step,
                       t->kv_cache, t->xattn_cache, t->normed);
    pico_vocab_project(t->model, t->normed, t->logits, 1);

    apply_repetition_penalty(t->logits, t->beam->tokens,
                             t->beam->n_tokens, REPETITION_PENALTY);
    apply_no_repeat_ngram(t->logits, t->beam->tokens,
                         t->beam->n_tokens, NO_REPEAT_NGRAM_SIZE);

    log_softmax_inplace(t->logits, VOCAB_SIZE);
    find_topk(t->logits, VOCAB_SIZE, BEAM_TOPK, t->tk, t->ts);
    
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    printf("=== PicoNLLB  (Beam Search, CTranslate2-style) ===\n\n");

    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s <model.safetensors> <src_lang> <tgt_lang> [token_ids...]\n"
            "  %s model.safetensors eng_Latn fra_Latn 59002 4 2\n", argv[0], argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int src_token = get_lang_token(argv[2]);
    int tgt_token = get_lang_token(argv[3]);
    if (src_token < 0 || tgt_token < 0) return 1;

    printf("[cfg] src=%s(%d) tgt=%s(%d) beam=%d rep=%.1f ngram=%d lp=%.1f\n",
           argv[2], src_token, argv[3], tgt_token,
           BEAM_SIZE, REPETITION_PENALTY, NO_REPEAT_NGRAM_SIZE, LENGTH_PENALTY);

    /* ── Load model ──────────────────────────────────────────────────────── */
    double t0 = now_ms();
    PicoModel model;
    if (pico_load(&model, model_path) != 0) return 1;
    printf("[load] %.0f ms\n", now_ms() - t0);

    /* ── Build source tokens ─────────────────────────────────────────────── */
    int src_tokens[MAX_SEQ_LEN];
    int n_src = 0;
    src_tokens[n_src++] = src_token;

    if (argc > 4) {
        for (int i = 4; i < argc && n_src < MAX_SEQ_LEN - 1; i++) {
            int tok = atoi(argv[i]);
            if (i == 4 && tok == src_token) continue;
            src_tokens[n_src++] = tok;
        }
    } else {
        src_tokens[n_src++] = 59002;
        src_tokens[n_src++] = 4;
        src_tokens[n_src++] = 2;
    }
    if (src_tokens[n_src - 1] != 2) src_tokens[n_src++] = 2;

    printf("[src] %d tokens:", n_src);
    for (int i = 0; i < n_src; i++) printf(" %d", src_tokens[i]);
    printf("\n");

    /* ── Allocate ────────────────────────────────────────────────────────── */
    float* encoder_out = (float*)malloc(n_src * D_MODEL * sizeof(float));
    float* beam_kv[2][BEAM_SIZE];
    for (int p = 0; p < 2; p++)
        for (int b = 0; b < BEAM_SIZE; b++)
            beam_kv[p][b] = (float*)calloc(KV_CACHE_TOTAL_FLOATS, sizeof(float));
    float* xattn_cache = (float*)calloc(XATTN_CACHE_TOTAL_FLOATS, sizeof(float));
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));

    /* ── Encode ──────────────────────────────────────────────────────────── */
    double t_enc = now_ms();
    pico_encode(&model, src_tokens, n_src, encoder_out);
    t_enc = now_ms() - t_enc;
    printf("[enc] %.0f ms\n", t_enc);

    /* ── Initialize thread pool ─────────────────────────────────────────────── */
    init_thread_pool();

    /* ── Beam search decode ──────────────────────────────────────────────── */
    int max_out = 3 * n_src + 10;
    if (max_out > MAX_GEN_LEN - 4) max_out = MAX_GEN_LEN - 4;

    double t_dec = now_ms();
    Beam beams[BEAM_SIZE];
    int cur_pool = 0;
    float normed[D_MODEL];

    for (int b = 0; b < BEAM_SIZE; b++) {
        beams[b].tokens[0] = 2;  // NLLB uses EOS (</s>) as decoder_start_token
        beams[b].n_tokens = 1;
        beams[b].score = 0.0f;
        beams[b].eos_score = 0.0f;
        beams[b].finished = 0;
    }

    /* Step 0: decoder_start_token (</s> = 2) -- projects cross-attention K/V once */
    pico_decode_forward(&model, encoder_out, n_src, 2, 0,
                        beam_kv[cur_pool][0], xattn_cache, normed);

    for (int b = 0; b < BEAM_SIZE; b++) {
        beams[b].tokens[1] = tgt_token;
        beams[b].n_tokens = 2;
        if (b > 0) memcpy(beam_kv[cur_pool][b], beam_kv[cur_pool][0],
                          KV_CACHE_TOTAL_FLOATS * sizeof(float));
    }

    /* Step 1: tgt_lang -- expand to BEAM_SIZE diverse beams */
    pico_decode_forward(&model, encoder_out, n_src, tgt_token, 1,
                        beam_kv[cur_pool][0], xattn_cache, normed);
    pico_vocab_project(&model, normed, logits, 1);

    /* Diagnostic: check logits for expected tokens before softmax */
    {
        /* Find max logit for numerical analysis */
        float max_logit = logits[0];
        int max_idx = 0;
        for (int v = 1; v < VOCAB_SIZE; v++) {
            if (logits[v] > max_logit) {
                max_logit = logits[v];
                max_idx = v;
            }
        }
        printf("[diag] max raw logit: %.4f at token %d\n", max_logit, max_idx);
        
        /* Hausa tokens for "Barka dai." */
        int diag[] = {3937, 162, 5288, 248075, 2, 77897, 39322, 248130};
        const char* dname[] = {"Bar","ka","dai",".","EOS","Yaya","kake","?"};
        printf("[diag] raw logits at step 1 (before softmax):\n");
        for (int d = 0; d < 8; d++)
            printf("  %s(%d): %.4f\n", dname[d], diag[d], logits[diag[d]]);
        
        /* After softmax, get log-probs */
        log_softmax_inplace(logits, VOCAB_SIZE);
        
        printf("[diag] log-probs at step 1 (after log-softmax):\n");
        for (int d = 0; d < 8; d++)
            printf("  %s(%d): %.4f\n", dname[d], diag[d], logits[diag[d]]);
        
        /* Check if log-softmax is correct */
        float log_sum_exp = logf(expf(max_logit - max_logit));  /* Should be close to 0 for max */
        printf("[diag] log-softmax check: max_logit=%.4f, log_prob[max]=%.4f\n", 
               max_logit, logits[max_idx]);
    }

    int   ib_tok[BEAM_SIZE];
    float ib_sc[BEAM_SIZE];
    find_topk(logits, VOCAB_SIZE, BEAM_SIZE, ib_tok, ib_sc);

    for (int b = 0; b < BEAM_SIZE; b++) {
        beams[b].tokens[2] = ib_tok[b];
        beams[b].n_tokens = 3;
        beams[b].score = ib_sc[b];
        beams[b].eos_score = 0.0f;
        beams[b].finished = (ib_tok[b] == 2);
        if (b > 0) memcpy(beam_kv[cur_pool][b], beam_kv[cur_pool][0],
                          KV_CACHE_TOTAL_FLOATS * sizeof(float));
    }
    printf("[beam] init:");
    for (int b = 0; b < BEAM_SIZE; b++) printf(" %d(%.1f)", ib_tok[b], ib_sc[b]);
    printf("\n");

    /* Completed hypotheses */
    Beam completed[BEAM_SIZE * 2];
    int n_completed = 0;
    /* CT2 doesn't do early exit when length_penalty=0 - let all beams finish naturally */
    int allow_early_exit = 0;  /* Disabled to match CT2 behavior */

    /* Steps 2..max_out: beam search with parallel beam processing */
    for (int step = 2; step < max_out + 2; step++) {
        Candidate cands[BEAM_SIZE * BEAM_TOPK + BEAM_SIZE];
        int nc = 0, n_active = 0;

        /* Allocate buffers for parallel processing */
        beam_task_t tasks[BEAM_SIZE];
        float normed_buf[BEAM_SIZE][D_MODEL];
        float logits_buf[BEAM_SIZE][VOCAB_SIZE];
        int tk_buf[BEAM_SIZE][BEAM_TOPK];
        float ts_buf[BEAM_SIZE][BEAM_TOPK];

        /* Count finished beams and add to candidates */
        for (int b = 0; b < BEAM_SIZE; b++) {
            if (beams[b].finished) {
                cands[nc++] = (Candidate){b, 2, beams[b].score};
            } else {
                n_active++;
            }
        }

        if (n_active == 0) break;

        /* Setup tasks for parallel beam processing */
        for (int b = 0; b < BEAM_SIZE; b++) {
            tasks[b] = (beam_task_t){
                .model = &model,
                .encoder_out = encoder_out,
                .n_src = n_src,
                .step = step,
                .kv_cache = beam_kv[cur_pool][b],
                .xattn_cache = xattn_cache,
                .normed = normed_buf[b],
                .logits = logits_buf[b],
                .beam = &beams[b],
                .tk = tk_buf[b],
                .ts = ts_buf[b]
            };
        }

        /* Dispatch to thread pool and wait */
        dispatch_beam_tasks(tasks, beams);

        /* Collect candidates from all beams */
        for (int b = 0; b < BEAM_SIZE; b++) {
            if (!beams[b].finished) {
                for (int k = 0; k < BEAM_TOPK; k++) {
                    cands[nc++] = (Candidate){b, tk_buf[b][k], beams[b].score + ts_buf[b][k]};
                }
            }
        }

        /* Sort descending by score */
        for (int i = 0; i < nc - 1; i++)
            for (int j = i + 1; j < nc; j++)
                if (cands[j].score > cands[i].score) {
                    Candidate t = cands[i]; cands[i] = cands[j]; cands[j] = t;
                }

        /* Select top BEAM_SIZE, reorder KV caches */
        int np = 1 - cur_pool;
        Beam nb[BEAM_SIZE];

        for (int i = 0; i < BEAM_SIZE; i++) {
            int par = cands[i].beam_idx;
            memcpy(nb[i].tokens, beams[par].tokens, beams[par].n_tokens * sizeof(int));
            nb[i].n_tokens = beams[par].n_tokens;
            nb[i].tokens[nb[i].n_tokens++] = cands[i].token;
            nb[i].score = cands[i].score;
            nb[i].finished = (cands[i].token == 2);
            
            /* Track EOS score separately (CT2 excludes it from final score) */
            if (cands[i].token == 2) {
                nb[i].eos_score = cands[i].score - beams[par].score;
            } else {
                nb[i].eos_score = beams[par].eos_score;
            }

            memcpy(beam_kv[np][i], beam_kv[cur_pool][par],
                   KV_CACHE_TOTAL_FLOATS * sizeof(float));

            if (nb[i].finished && n_completed < BEAM_SIZE * 2)
                memcpy(&completed[n_completed++], &nb[i], sizeof(Beam));
        }
        memcpy(beams, nb, sizeof(beams));
        cur_pool = np;

        printf("\r[beam] step %d  active=%d  done=%d  len=%d  score=%.1f  ",
               step, n_active, n_completed, beams[0].n_tokens - 2, beams[0].score);
        fflush(stdout);

        /* CTranslate2 early exit: when length_penalty==0, stop as soon as
         * top beam finishes AND we have at least 1 completed hypothesis */
        if (allow_early_exit && n_completed > 0 && beams[0].finished) {
            printf("\n[beam] early exit (top beam finished)\n");
            break;
        }

        /* General early stop: all beams finished */
        if (n_active == 0) break;
    }
    printf("\n");
    t_dec = now_ms() - t_dec;

    /* ── Add remaining finished beams ────────────────────────────────────── */
    for (int b = 0; b < BEAM_SIZE; b++)
        if (beams[b].finished && n_completed < BEAM_SIZE * 2)
            memcpy(&completed[n_completed++], &beams[b], sizeof(Beam));

    if (n_completed == 0) {
        printf("[beam] WARNING: no EOS generated, using best active beam\n");
        memcpy(&completed[0], &beams[0], sizeof(Beam));
        n_completed = 1;
    }

    /* ── Pick best hypothesis ────────────────────────────────────────────── */
    int bi = 0;
    float bn = -1e30f;
    for (int c = 0; c < n_completed; c++) {
        int cl = completed[c].n_tokens - 2;
        /* Exclude EOS score from final score (match CT2 behavior) */
        float final_score = completed[c].score - completed[c].eos_score;
        /* CT2 always uses normalized score for selection, regardless of length_penalty */
        float ns = final_score / (float)(cl + 1);  /* Normalize by length including EOS */
        if (ns > bn) { bn = ns; bi = c; }
    }
    Beam* best = &completed[bi];
    int n_out = best->n_tokens - 2;
    if (best->tokens[best->n_tokens - 1] == 2) n_out--;
    float best_score = best->score - best->eos_score;  /* Report score without EOS */

    printf("\n=== RESULTS ===\n");
    for (int c = 0; c < n_completed; c++) {
        int cl = completed[c].n_tokens - 2;
        float final_score = completed[c].score - completed[c].eos_score;
        /* CT2 normalizes score by target length (including EOS in count) */
        float normalized_score = final_score / (float)(cl + 1);  /* +1 for EOS */
        printf("  hyp%d: score=%.2f (norm=%.2f) len=%d [", c, final_score, normalized_score, cl);
        for (int i = 2; i < completed[c].n_tokens; i++)
            printf("%d%s", completed[c].tokens[i], i < completed[c].n_tokens - 1 ? " " : "");
        printf("]\n");
    }

    double tps = n_out > 0 ? (double)n_out / (t_dec / 1000.0) : 0;
    /* Report normalized score like CT2 - use same calculation as hypotheses */
    int best_cl = best->n_tokens - 2;
    float normalized_best_score = best_score / (float)(best_cl + 1);  /* +1 for EOS */
    printf("\nbest: %d tokens, score=%.2f, enc=%.0fms, dec=%.0fms, %.2f tok/s\n",
           n_out, normalized_best_score, t_enc, t_dec, tps);
    printf("tokens:");
    /* Output tokens without EOS (match CT2 format) */
    for (int i = 2; i < best->n_tokens; i++) {
        if (best->tokens[i] == 2) break;  /* Stop at EOS, don't include it */
        printf(" %d", best->tokens[i]);
    }
    printf("\n");

    /* Python decode command */
    printf("\npython3 -c \"from transformers import AutoTokenizer; "
           "t=AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M'); "
           "print(t.decode([");
    int first = 1;
    for (int i = 2; i < best->n_tokens; i++) {
        if (best->tokens[i] == 2) continue;
        printf("%s%d", first ? "" : ",", best->tokens[i]);
        first = 0;
    }
    printf("]))\"\n");

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    shutdown_thread_pool();
    
    free(encoder_out);
    for (int p = 0; p < 2; p++)
        for (int b = 0; b < BEAM_SIZE; b++) free(beam_kv[p][b]);
    free(xattn_cache);
    free(logits);
    pico_free(&model);
    return 0;
}
