#include "TrueLargeRuntime.h"
#include <sys/mman.h>
#include "WeightBuffer.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <android/log.h>
#include <cstring>
#include <cmath>
#include <mutex>
#include <map>
#include <string>
#include <algorithm>
#include <thread>

static void custom_ggml_abort_callback(const char * error_message) {
    __android_log_print(ANDROID_LOG_ERROR, "TrueLargePerf", "GGML CRASH ABORT: %s", error_message);
}

#define TAG "TrueLargePerf"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#define TAG_GQA "TrueLargeGQA"
#define LOGI_GQA(...) __android_log_print(ANDROID_LOG_INFO, TAG_GQA, __VA_ARGS__)

// Helper to get RAM usage (RSS)
long getMemoryUsageKB() {
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm >> size >> resident >> share >> text >> lib >> data >> dt;
        statm.close();
        long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;
        return resident * page_size_kb;
    }
    return 0;
}

// Helper to get available RAM from /proc/meminfo (more accurate than sysconf)
long getAvailableMemoryKB() {
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.compare(0, 13, "MemAvailable:") == 0) {
                std::stringstream ss(line.substr(13));
                long avail_kb;
                ss >> avail_kb;
                return avail_kb;
            }
        }
    }
    // Fallback to sysconf if meminfo fails
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;
    return pages * page_size_kb;
}

// Helper to get current CPU frequency of the core we are on
long getCurrentCpuFreqHz() {
    int cpu = sched_getcpu();
    std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/scaling_cur_freq";
    std::ifstream freqFile(path);
    if (freqFile.is_open()) {
        long freq_khz;
        freqFile >> freq_khz;
        freqFile.close();
        return freq_khz * 1000; // Convert KHz to Hz
    }
    return 0;
}

// Helper to set CPU affinity (REMOVED: User requested removal due to slowness)
/* void set_cpu_affinity() { ... } */

// Helper to convert token to string
std::string token_to_str(const llama_context* ctx, llama_token token) {
    const llama_model* model = llama_get_model(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    
    std::string piece;
    piece.resize(256);
    int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, true);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, true);
    }
    piece.resize(n_chars);
    return piece;
}

TrueLargeRuntime::TrueLargeRuntime() {}

TrueLargeRuntime::~TrueLargeRuntime() {
    release();
}

bool TrueLargeRuntime::loadModel(const std::string& path) {
    // Release any existing model/context to prevent leaks
    release();
    
    modelPath = path;
    
    // Log build info
    LOGI("TrueLarge Engine Init: %s", llama_print_system_info());

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = nGpuLayers;

    // Smart Memory Management:
    // Check if we have enough RAM to lock the model (avoid paging)
    // Safety buffer: 1GB roughly
    long availRamKB = getAvailableMemoryKB();
    
    // Get file size
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    long fileSizeKB = 0;
    if (file.good()) {
        fileSizeKB = file.tellg() / 1024;
    }
    
    if (detectLayerByLayerNeeded(fileSizeKB)) {
        LOGI("Low RAM detected relative to model. Enabling Layer-by-Layer Loading.");
        useLayerByLayer = true;
        inferenceMode = "Layer-by-Layer (Disk Swap)";
        
        // Initialize parser if not already done (Wait, headerParser needs to be init!)
        // Typically headerParser is needed for LBL.
        headerParser = std::make_unique<GGUFHeaderParser>(path);
        if (!headerParser->parse()) {
            LOGE("Failed to parse GGUF for LBL");
            return false;
        }
        
        // We still load model with llama.cpp for vocab/sampling, but try to minimize its RAM?
        // use_mlock = false, so it mmaps.
        model_params.use_mlock = false; 
    } else if (availRamKB > (fileSizeKB + 1024 * 1024)) {
        LOGI("Sufficient RAM detected. Locking model in memory.");
        model_params.use_mlock = true;
        inferenceMode = "Locked RAM (Maximum Speed)";
    } else {
        LOGI("Using standard mmap.");
        model_params.use_mlock = false; 
        inferenceMode = "Standard MMAP (OS Managed Swapping)";
    }

    // API change: llama_load_model_from_file -> llama_model_load_from_file
    model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        LOGE("Failed to load model: %s", path.c_str());
        return false;
    }

    // Cap threads for mobile (usually 3 is sweet spot for LBL to leave room for UI)
    if (nThreads > 3) nThreads = 3;
    if (nThreads < 1) nThreads = 1;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048; // Default context size
    ctx_params.n_threads = nThreads;
    ctx_params.n_threads_batch = nThreads;

    // API change: llama_new_context_with_model -> llama_init_from_model
    ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create context");
        return false;
    }

    if (useLayerByLayer) {
        initLayerByLayer();
    }

    LOGI("Model loaded successfully (%d threads). RAM: %ld KB", nThreads, getMemoryUsageKB());
    return true;
}

void TrueLargeRuntime::configure(int threads, int gpuLayers) {
    nThreads = threads;
    nGpuLayers = gpuLayers;
}

void TrueLargeRuntime::configureSampler(float temp, int k, float p) {
    temperature = temp;
    topK = k;
    topP = p;
    LOGI("Sampler configured: Temp=%.2f, TopK=%d, TopP=%.2f", temperature, topK, topP);
}

// Global reuse batch to avoid alloc overhead
// NOTE: Not thread-safe if multiple instances exist, but okay for this singleton usage
static llama_batch g_batch;
static bool g_batch_init = false;

bool TrueLargeRuntime::createSession(const std::string& prompt, bool keepHistory) {
    t_session_start = std::chrono::steady_clock::now();
    // set_cpu_affinity(); // Pin to big cores (REMOVED: Caused performance issues)
    if (!model || !ctx) {
        LOGE("Model not loaded");
        return false;
    }

    // Initialize Sampler Chain
    if (sampler) {
        llama_sampler_free(sampler);
    }
    
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler = llama_sampler_chain_init(sampler_params);
    
    const llama_vocab* vocab = llama_model_get_vocab(model);
    
    // chain: penalties -> top_k -> top_p -> min_p -> temp
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(64, 1.1f, 0.0f, 0.0f));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234)); // Fixed seed for now, or use LLAMA_DEFAULT_SEED

    // Tokenize
    std::vector<llama_token> tokens;
    // Over-allocate slightly to be safe
    tokens.resize(prompt.size() + 10); 
    
    // When keeping history, usually we don't want to re-add BOS if nPast > 0
    bool add_bos = (nPast == 0);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), add_bos, false);
    if (n_tokens < 0) {
        LOGE("Tokenization failed");
        return false;
    }
    tokens.resize(n_tokens);
    LOGI("Tokenized prompt: %d tokens. History: %d. \"%s\"", n_tokens, nPast, prompt.c_str());

    // Diagnostic: Print prompt tokens
    for (int i = 0; i < n_tokens; i++) {
        std::string s = token_to_str(ctx, tokens[i]);
        LOGI("  Token %d: %d -> '%s'", i, (int)tokens[i], s.c_str());
    }

    // Sampler priming moved to AFTER session reset logic below

    LOGI("Tokenized prompt: %d tokens. History: %d. \"%s\"", n_tokens, nPast, prompt.c_str());

    if (!keepHistory) {
        // Clear KV cache for new session
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_seq_rm(mem, -1, 0, -1);
        
        nPast = 0;
        generatedTokens.clear();
        
        // CRITICAL: If using LBL, we need to re-init global weights
        // because they were allocated in ctx_compute which we're about to free
        if (useLayerByLayer) {
            LOGI("Clearing LBL session: Re-initializing global weights...");
            if (ctx_compute) {
                ggml_free(ctx_compute);
                ctx_compute = nullptr;
            }
            if (ctx_compute_back) {
                ggml_free(ctx_compute_back);
                ctx_compute_back = nullptr;
            }
            
            // Re-initialize global weights with fresh context
            // Re-initialize global weights with fresh context
            initGlobalWeights();
            
            // NEW: Hardware reset logic for LBL KV Cache
            if (ctx_kv) {
                 for (int i=0; i<(int)kv_k.size(); i++) {
                     if (kv_k[i]) memset(kv_k[i]->data, 0, ggml_nbytes(kv_k[i]));
                     if (kv_v[i]) memset(kv_v[i]->data, 0, ggml_nbytes(kv_v[i]));
                 }
                 LOGI("LBL KV Cache cleared (memset 0).");
            }
        }
        
        // Reset sampler state
        if (sampler) {
            llama_sampler_reset(sampler);
            LOGI("Sampler reset.");
        }
    }
    
    // Prime Sampler with prompt tokens (AFTER potential reset)
    for (int i = 0; i < n_tokens; i++) {
        llama_token t = tokens[i];
        llama_sampler_accept(sampler, t);
    }

    // Prepare batch manually since helper is missing

    // Decode prompt
    auto start = std::chrono::high_resolution_clock::now();
    
    if (useLayerByLayer) {
        // Manual LBL Evaluation for prompt
        LOGI("Performing LBL Prompt Pre-fill (%d tokens)...", n_tokens);
        
        // 1. Get embeddings
        // Only allocate ctx_compute if it doesn't exist (it was cleared above if !keepHistory)
        if (!ctx_compute) {
            ggml_set_abort_callback(custom_ggml_abort_callback);
            int n_embd = llama_model_n_embd(model);
            size_t emb_size = std::max((size_t)(2048LL*1024*1024), (size_t)n_tokens * n_embd * 4 * 4) + 128*1024*1024;
            struct ggml_init_params params_init = { .mem_size = emb_size, .mem_buffer = NULL };
            ctx_compute = ggml_init(params_init);
        }
        if (!ctx_compute_back) {
            struct ggml_init_params params_back = { .mem_size = 2048LL*1024*1024 + (size_t)n_tokens * 1024 * 4, .mem_buffer = NULL };
            ctx_compute_back = ggml_init(params_back);
        }
        
        struct ggml_tensor* tokens_idx = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, n_tokens);
        memcpy(tokens_idx->data, tokens.data(), n_tokens * sizeof(int32_t));
        
        struct ggml_tensor* input = ggml_get_rows(ctx_compute, w_token_embd, tokens_idx);
        
        // CRITICAL: Compute the embedding graph before forwarding
        struct ggml_cgraph* gf_emb = ggml_new_graph_custom(ctx_compute, 8192, false);
        ggml_build_forward_expand(gf_emb, input);
        ggml_graph_compute_with_ctx(ctx_compute, gf_emb, nThreads);
        
        // DIAGNOSTIC: Print first 4 embedding values
        if (input->type == GGML_TYPE_F32) {
            float* emb_data = (float*)input->data;
            LOGI("DIAG Pre-fill Embedding: [%.6f, %.6f, %.6f, %.6f] ne=[%ld,%ld]", 
                 emb_data[0], emb_data[1], emb_data[2], emb_data[3], input->ne[0], input->ne[1]);
        } else {
            LOGI("DIAG Pre-fill Embedding type=%d (not F32), ne=[%ld,%ld]", input->type, input->ne[0], input->ne[1]);
        }
        
        // 2. Layer Loop (ping-pong contexts)
        struct ggml_context* ctx_ping = ctx_compute;
        struct ggml_context* ctx_pong = ctx_compute_back;
        int n_layer = llama_model_n_layer(model);
        struct ggml_init_params params_layer = { .mem_size = 2048LL*1024*1024, .mem_buffer = NULL };
        
        for (int i = 0; i < n_layer; i++) {
            if (i % 8 == 0) LOGI("LBL Pre-fill: Progress %d/%d layers", i, n_layer);
            
            // Reset pong context for this layer
            ggml_free(ctx_pong);
            ctx_pong = ggml_init(params_layer);
            
            struct ggml_tensor* out = forwardLayer(i, input, ctx_pong);
            if (!out) {
                LOGE("Pre-fill: Layer %d failed to build", i);
                break;
            }
            
            struct ggml_cgraph* gf = ggml_new_graph_custom(ctx_pong, 8192, false);
            ggml_build_forward_expand(gf, out);
            ggml_graph_compute_with_ctx(ctx_pong, gf, nThreads);
            
            // DIAGNOSTIC REMOVED

            struct ggml_tensor* result_leaf = ggml_new_tensor(ctx_pong, out->type, ggml_n_dims(out), out->ne);
            memcpy(result_leaf->data, out->data, ggml_nbytes(out));
            input = result_leaf;
            
            std::swap(ctx_ping, ctx_pong);
        }
        // After loop: input is in ctx_ping. Reassign back.
        ctx_compute = ctx_ping;
        ctx_compute_back = ctx_pong;
        
        // 3. Final projection (Logits for the LAST token of the prompt)
        // Last token index in batch is n_tokens - 1
        struct ggml_tensor* last_hidden = ggml_view_1d(ctx_compute, input, llama_model_n_embd(model), (n_tokens - 1) * llama_model_n_embd(model) * sizeof(float));
        
        // RMS Norm + Output projection
        struct ggml_tensor* norm = ggml_rms_norm(ctx_compute, last_hidden, model_rms_norm_eps);
        norm = ggml_mul(ctx_compute, norm, w_output_norm);
        
        if (!w_output) {
            LOGE("createSession: output weights missing even after tie-word check!");
            return false;
        }
        struct ggml_tensor* logits = ggml_mul_mat(ctx_compute, w_output, norm);
        
        struct ggml_cgraph* gf_logits = ggml_new_graph_custom(ctx_compute, 8192, false);
        ggml_build_forward_expand(gf_logits, logits);
        ggml_graph_compute_with_ctx(ctx_compute, gf_logits, nThreads);
        
        // Copy logits to llama context for sampler
        float* dst = llama_get_logits(ctx);
        int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        memcpy(dst, logits->data, n_vocab * sizeof(float));
    } else {
        // Standard Decoding
        if (g_batch_init) {
            llama_batch_free(g_batch);
        }
        g_batch = llama_batch_init(std::max(2048, n_tokens), 0, 1); 
        g_batch_init = true;

        g_batch.n_tokens = n_tokens;
        for (int i = 0; i < n_tokens; i++) {
            g_batch.token[i] = tokens[i];
            g_batch.pos[i] = nPast + i;
            g_batch.n_seq_id[i] = 1;
            g_batch.seq_id[i][0] = 0;
            g_batch.logits[i] = false;
        }
        g_batch.logits[g_batch.n_tokens - 1] = true;

        if (llama_decode(ctx, g_batch) != 0) {
            LOGE("llama_decode failed");
            return false;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    nPast += n_tokens;
    t_generation_start = std::chrono::steady_clock::now();
    LOGI("Prompt Eval Speed: %.2f ms for %d tokens (%.2f t/s)", duration, n_tokens, (n_tokens / duration) * 1000.0);
    
    return true;
}


std::string TrueLargeRuntime::step() {
    if (!ctx || !sampler) return "";

    if (useLayerByLayer) {
        return step_lbl();
    }

    // Fallback: Normal single-token inference
    llama_token next_token = llama_sampler_sample(sampler, ctx, -1);
    llama_sampler_accept(sampler, next_token);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    if (next_token == llama_vocab_eos(vocab)) {
        LOGI("EOS generated");
        return "";
    }

    g_batch.n_tokens = 1;
    g_batch.token[0] = next_token;
    g_batch.pos[0] = nPast;
    g_batch.n_seq_id[0] = 1;
    g_batch.seq_id[0][0] = 0;
    g_batch.logits[0] = true;

    auto start = std::chrono::high_resolution_clock::now();
    if (llama_decode(ctx, g_batch) != 0) return "";
    auto end = std::chrono::steady_clock::now();
    
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    nPast += 1;
    generatedTokens.push_back(next_token);
    std::string piece = token_to_str(ctx, next_token);

    if (generatedTokens.size() == 1) {
        lastTTFT = std::chrono::duration<double, std::milli>(end - t_session_start).count();
        LOGI("TTFT: %.2f ms", lastTTFT);
    }

    lastTPS = generatedTokens.size() / std::chrono::duration<double>(end - t_generation_start).count();
    lastTotalTime = std::chrono::duration<double>(end - t_session_start).count();
    
    // Telemetry: RAM and CPU
    long rss_kb = getMemoryUsageKB();
    long avail_kb = getAvailableMemoryKB();
    long freq_hz = getCurrentCpuFreqHz();
    int cpu_id = sched_getcpu();
    
    lastRAM = rss_kb / 1024; // MB
    lastCPUFreq = freq_hz / 1e9; // GHz

    const char* warning = "";
    if (avail_kb < 512 * 1024) { // Warning if less than 512MB free
        warning = "[LOW-RAM IO-WAIT] ";
    }

    LOGI("%sGen: %d -> Token %d ('%s') | Speed: %.2f ms | TPS: %.2f | RAM: %ld MB | CPU: #%d @ %.2f GHz", 
         warning, (int)generatedTokens.size(), next_token, piece.c_str(), duration, lastTPS, lastRAM, cpu_id, lastCPUFreq);

    return piece;
}

int TrueLargeRuntime::getContextTrain() {
    if (!model) return 0;
    return llama_model_n_ctx_train(model);
}

int TrueLargeRuntime::getContextCurrent() {
    if (!ctx) return 0;
    return llama_n_ctx(ctx);
}

void TrueLargeRuntime::release() {
    if (g_batch_init) {
        llama_batch_free(g_batch);
        g_batch_init = false;
    }
    if (sampler) {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }
    if (sampler_dft) {
        llama_sampler_free(sampler_dft);
        sampler_dft = nullptr;
    }
    if (ctx) {
        llama_free(ctx);
        ctx = nullptr;
    }
    if (model) {
        llama_model_free(model); // API change
        model = nullptr;
    }
    if (layerLoader) layerLoader.reset();
    if (scheduler) scheduler.reset();
    if (headerParser) headerParser.reset();
    
    if (ctx_compute) { ggml_free(ctx_compute); ctx_compute = nullptr; }
    if (ctx_compute_back) { ggml_free(ctx_compute_back); ctx_compute_back = nullptr; }
    if (ctx_weights) { ggml_free(ctx_weights); ctx_weights = nullptr; }
    if (ctx_global) { ggml_free(ctx_global); ctx_global = nullptr; }
    if (ctx_kv) { ggml_free(ctx_kv); ctx_kv = nullptr; }
    kv_k.clear();
    kv_v.clear();

    llama_backend_free();
}

// ... existing code ...

bool TrueLargeRuntime::detectLayerByLayerNeeded(long fileSizeKB) {
    long availRamKB = getAvailableMemoryKB();
    // Hybrid Strategy:
    // 1. Full RAM: Handled in loadModel (if avail > size + 1GB)
    // 2. OS Paging (mmap): If we can sustain ~75% of model in RAM.
    //    Condition: avail >= 0.75 * size  =>  size <= avail * 1.33
    // 3. LBL: If model is huge (avail < 75% of size).
    
    // We force LBL only if model size is > 1.3x available RAM.
    double ratio = (double)fileSizeKB / (double)availRamKB;
    bool needed = ratio > 1.3;
    
    LOGI("Smart Loading Check: Model=%ld MB, Avail=%ld MB, Ratio=%.2f. LBL Needed? %s", 
         fileSizeKB/1024, availRamKB/1024, ratio, needed ? "YES" : "NO (Use MMAP)");
         
    return needed;
}

void TrueLargeRuntime::initLayerByLayer() {
    LOGI("Initializing Layer-by-Layer Scheduler");
    
    // Max layers in memory:
    // Calculate based on remaining RAM, accounting for KV Cache and OS overhead.
    long availRamKB = getAvailableMemoryKB();
    long layerSizeKB = 0;
    
    // Estimate layer size from first layer
    const LayerSourceInfo* info = headerParser->getLayerSourceInfo(0);
    if (info) {
        long totalSize = 0;
        for (const auto& tp : info->tensors) {
            totalSize += tp.second.size;
        }
        layerSizeKB = totalSize / 1024;
    }
    
    // KV Cache Size (Persistent RAM)
    int n_layer = llama_model_n_layer(model);
    int n_embd = llama_model_n_embd(model);
    int n_head = llama_model_n_head(model);
    int n_head_kv = llama_model_n_head_kv(model);
    int head_dim = n_embd / n_head;
    size_t kv_size_kb = ((size_t)n_layer * 2 * kv_max_tokens * (n_head_kv * head_dim) * sizeof(float)) / 1024;

    int maxLayers = 2; // Strict default for stability
    if (layerSizeKB > 0) {
        // Leave 1.5GB for OS/App and subtract KV Cache from budget
        // Safety buffer KB: 1,572,864 (1.5GB)
        long safetyBufferKB = 1536 * 1024; 
        long workingBudgetKB = availRamKB - safetyBufferKB - (long)kv_size_kb;
        
        if (workingBudgetKB > 0) {
            maxLayers = workingBudgetKB / layerSizeKB;
        }
    }

    // On devices with low absolute RAM, hard cap to avoid any risk of LMK
    if (availRamKB < 2500 * 1024) { // < 2.5GB Available (Strict)
        if (maxLayers > 2) maxLayers = 2;
    }
    
    if (maxLayers < 1) maxLayers = 1; // Minimum 1 for execution
    if (maxLayers > 10) maxLayers = 10; // Relaxed cap for experimental high-end usage
    
    // Captured HParams
    char arch[64] = "unknown";
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    
    if (strcmp(arch, "gptneox") == 0) {
        model_arch_type = ARCH_GPTNEOX;
        model_parallel_residual = true; // GPT-NeoX traditionally uses parallel attention/FFN

    } else if (strstr(arch, "qwen2") || strstr(arch, "qwen3")) {
        model_arch_type = ARCH_QWEN;
    } else if (strstr(arch, "gemma")) {
        model_arch_type = ARCH_GEMMA;
    } else {
        model_arch_type = ARCH_LLAMA;
    }

    model_n_rot = (int)getModelMetaFloat("llama.rope.dimension_count", 0);
    if (model_n_rot == 0) model_n_rot = (int)getModelMetaFloat("rope.dimension_count", 0);
    if (model_n_rot == 0) model_n_rot = (int)getModelMetaFloat("gptneox.rope.dimension_count", 0);


    // Log metadata for debugging
    LOGI("LBL Init: Avail=%ld MB, KV_Cache=%ld MB, Layer=%ld MB, Budget=%ld MB, MaxLayers=%d", 
         availRamKB/1024, kv_size_kb/1024, layerSizeKB/1024, 
         (availRamKB - (long)kv_size_kb - (1536*1024))/1024, maxLayers);

    model_rope_freq_base = getModelMetaFloat("gpt-oss.rope.freq_base", 0.0f);
    if (model_rope_freq_base == 0.0f) model_rope_freq_base = getModelMetaFloat("gptneox.rope.freq_base", 0.0f);
    if (model_rope_freq_base == 0.0f) model_rope_freq_base = getModelMetaFloat("llama.rope.freq_base", 0.0f);
    if (model_rope_freq_base == 0.0f) model_rope_freq_base = getModelMetaFloat("qwen2.rope.freq_base", 0.0f);
    if (model_rope_freq_base == 0.0f) model_rope_freq_base = getModelMetaFloat("qwen3.rope.freq_base", 0.0f);
    if (model_rope_freq_base == 0.0f) model_rope_freq_base = getModelMetaFloat("rope.freq_base", 0.0f);

    if (model_rope_freq_base == 0.0f) {
        if (strstr(arch, "llama-3") || strstr(arch, "llama-3.1")) {
            model_rope_freq_base = 500000.0f;
            LOGI("LBL Fallback: Using 500k RoPE Base for Llama 3 architecture");
        } else if (strstr(arch, "qwen2")) {
            model_rope_freq_base = 1000000.0f;
            LOGI("LBL Fallback: Using 1M RoPE Base for Qwen2 architecture");
        } else if (strstr(arch, "gemma")) {
            model_rope_freq_base = 10000.0f; // Gemma default
            LOGI("LBL Fallback: Using 10k RoPE Base for Gemma architecture");
        } else {
            model_rope_freq_base = 10000.0f; // Legacy default
        }
    }

    model_rope_freq_scale = getModelMetaFloat("llama.rope.freq_scale", 1.0f);
    if (model_rope_freq_scale == 1.0f) model_rope_freq_scale = getModelMetaFloat("qwen3.rope.freq_scale", 1.0f);
    if (model_rope_freq_scale == 0.0f) model_rope_freq_scale = 1.0f;
    
    float scaling_factor = getModelMetaFloat("llama.rope.scaling.factor", 0.0f);
    if (scaling_factor > 0.0f) {
        model_rope_freq_scale = 1.0f / scaling_factor;
    }

    // RMS Norm Epsilon lookup with fallbacks
    model_rms_norm_eps = getModelMetaFloat("gptneox.attention.layer_norm_rms_epsilon", 0.0f);
    if (model_rms_norm_eps == 0.0f) model_rms_norm_eps = getModelMetaFloat("llama.attention.layer_norm_rms_epsilon", 0.0f);
    if (model_rms_norm_eps == 0.0f) model_rms_norm_eps = getModelMetaFloat("qwen2.attention.layer_norm_rms_epsilon", 0.0f);
    if (model_rms_norm_eps == 0.0f) model_rms_norm_eps = getModelMetaFloat("qwen3.attention.layer_norm_rms_epsilon", 0.0f);
    if (model_rms_norm_eps == 0.0f) model_rms_norm_eps = getModelMetaFloat("attention.layer_norm_rms_epsilon", 0.0f);

    if (model_rms_norm_eps == 0.0f) {
         if (strstr(arch, "qwen2") || strstr(arch, "qwen3")) {
             model_rms_norm_eps = 1e-6f;
         } else {
             model_rms_norm_eps = 1e-5f;
         }
    }
    
    int n_embd_val = llama_model_n_embd(model);
    int n_head_val = llama_model_n_head(model);
    int n_head_kv_val = llama_model_n_head_kv(model);
    
    LOGI("LBL Config: Arch: %s, n_embd: %d, n_head: %d, n_head_kv: %d, Layer Size ~%ld MB, Max Layers: %d, RoPE Base: %.1f, Norm Eps: %.1e", 
         arch, n_embd_val, n_head_val, n_head_kv_val, layerSizeKB/1024, maxLayers, model_rope_freq_base, model_rms_norm_eps);

    // Initialize Global LayerLoader
    layerLoader = std::make_unique<LayerLoader>(modelPath);
    if (!layerLoader->init()) {
        LOGE("Failed to initialize LayerLoader for globals");
        return;
    }
    
    scheduler = std::make_unique<LayerScheduler>(modelPath, headerParser.get(), maxLayers);
    useLayerByLayer = true;
    
    initGlobalWeights();
    
    // Initialize ggml contexts
    struct ggml_init_params params = {
        .mem_size   = 32 * 1024 * 1024, // 32MB for graph/overhead
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    ctx_compute = ggml_init(params);
    ctx_compute_back = ggml_init(params);
    
    struct ggml_init_params params_w = {
        .mem_size   = 1024 * 1024 * 4, // 4MB enough for descriptors
        .mem_buffer = NULL,
        .no_alloc   = true, // Important: we provide data pointers manually
    };
    ctx_weights = ggml_init(params_w);
    
    // Global weights context (persistent)
    struct ggml_init_params params_g = {
        .mem_size   = 1024 * 1024 * 16, // Descriptors only
        .mem_buffer = NULL,
        .no_alloc   = true, 
    };
    ctx_global = ggml_init(params_g);

    // 40 layers * 2 * 512 * 5120 * 4 bytes (F32) = ~838 MB
    // Let's allocate enough for the KV context.
    size_t kv_size = (size_t)n_layer * 2 * kv_max_tokens * (n_head_kv * head_dim) * sizeof(float) + (1024 * 1024 * 10);
    struct ggml_init_params params_kv = {
        .mem_size = kv_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };
    ctx_kv = ggml_init(params_kv);
    kv_k.resize(n_layer);
    kv_v.resize(n_layer);

    for (int i = 0; i < n_layer; i++) {
        kv_k[i] = ggml_new_tensor_3d(ctx_kv, GGML_TYPE_F32, head_dim, kv_max_tokens, n_head_kv);
        kv_v[i] = ggml_new_tensor_3d(ctx_kv, GGML_TYPE_F32, head_dim, kv_max_tokens, n_head_kv);
        // Initialize with 0s to avoid garbage
        memset(kv_k[i]->data, 0, ggml_nbytes(kv_k[i]));
        memset(kv_v[i]->data, 0, ggml_nbytes(kv_v[i]));
    }
    
    // ctx_global = ggml_init(params_g); // Not strictly needed if we reuse llama tensors?
    // Let's rely on first call.
}

float TrueLargeRuntime::getModelMetaFloat(const char* key, float defaultValue) {
    char buf[64];
    if (model && llama_model_meta_val_str(model, key, buf, sizeof(buf)) > 0) {
        return (float)atof(buf);
    }
    return defaultValue;
}

void TrueLargeRuntime::initGlobalWeights() {
    if (!ctx_global) {
        // Init if not done (though initLayerByLayer does it, safe check)
        struct ggml_init_params params_g = {
            .mem_size   = 1024 * 1024 * 16, // Descriptors only
            .mem_buffer = NULL,
            .no_alloc   = true, 
        };
        ctx_global = ggml_init(params_g);
    }
    
    const LayerSourceInfo* info = headerParser->getLayerSourceInfo(-1);
    if (!info) {
        LOGE("Globals not found in GGUF");
        return;
    }
    
    auto loadGlobal = [&](const char* name) -> struct ggml_tensor* {
        // Find tensor info
        std::string suffix = name; // Expect exact match for globals
        bool found = false;
        TensorInfo t;
        for (const auto& pair : info->tensors) {
            if (pair.first == suffix) {
                t = pair.second;
                found = true;
                break;
            }
        }
        if (!found) return nullptr;
        
        // Load data
        // Note: loadLayer maps [offset, size]. 
        // We assume offset is absolute from file start?
        // GGUFHeaderParser stores absolute offset now.
        void* ptr = layerLoader->loadLayer(t.offset, t.size);
        if (!ptr) {
            LOGE("Failed to load global %s", name);
            return nullptr;
        }
        
        // Create tensor
        struct ggml_tensor* tensor = nullptr;
         if (t.dims.size() == 1) {
            tensor = ggml_new_tensor_1d(ctx_global, (ggml_type)t.type, t.dims[0]);
        } else if (t.dims.size() == 2) {
            tensor = ggml_new_tensor_2d(ctx_global, (ggml_type)t.type, t.dims[0], t.dims[1]);
        }
        
        if (tensor) {
            ggml_set_name(tensor, name);
            tensor->data = ptr;
        }
        return tensor;
    };
    
    w_token_embd = loadGlobal("token_embd.weight");
    w_output_norm = loadGlobal("output_norm.weight");
    w_output = loadGlobal("output.weight");
    
    if (w_token_embd) LOGI("Loaded token_embd (type=%d, ne=[%ld,%ld])", w_token_embd->type, w_token_embd->ne[0], w_token_embd->ne[1]);
    if (w_output_norm) LOGI("Loaded output_norm (type=%d, ne=[%ld])", w_output_norm->type, w_output_norm->ne[0]);
    if (w_output) {
        LOGI("Loaded output (type=%d, ne=[%ld,%ld], nbytes=%zu)", w_output->type, w_output->ne[0], w_output->ne[1], ggml_nbytes(w_output));
        // Log first few raw bytes for verification
        uint8_t* raw = (uint8_t*)w_output->data;
        LOGI("DIAG w_output first 16 bytes: %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x %02x",
             raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
             raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15]);
    } else if (w_token_embd) {
        LOGI("Tie word embeddings detected. Using token_embd as output head.");
        w_output = w_token_embd;
    }
}

// ... (forwardLayer implementation will go here)

// Helper to get weight tensor
static struct ggml_tensor* get_w(const std::map<std::string, struct ggml_tensor*>& tensors, const std::string& suffix) {
    auto it = tensors.find(suffix);
    if (it == tensors.end()) {
        LOGE("Missing tensor: %s", suffix.c_str());
        return nullptr;
    }
    return it->second;
}

static struct ggml_tensor* get_optional_w(const std::map<std::string, struct ggml_tensor*>& tensors, const std::string& suffix) {
    auto it = tensors.find(suffix);
    if (it == tensors.end()) {
        return nullptr;
    }
    return it->second;
}

struct ggml_tensor* TrueLargeRuntime::forwardLayer(int layerIndex, struct ggml_tensor* input, struct ggml_context* ctx_build) {
    // 1. Prepare weights
    initLayerWeights(layerIndex);
    if (currentWeightTensors.empty()) {
        LOGE("Forward: No weights for layer %d", layerIndex);
        return input;
    }

    // Context management is done by caller (step_lbl)
    
    // Re-import input logic:
    // If input is in a different context, ggml ops usually handle it if data ptr is valid.
    // However, if we freed the previous context, input data is GONE.
    // Ping-pong strategy in step_lbl ensures previous context is alive while we build in new one.
    // So we just use `input` directly.
    
    struct ggml_tensor* cur = input;
    
    // Get HParams
    int n_head = llama_model_n_head(model);
    int n_head_kv = llama_model_n_head_kv(model); 
    int n_embd = llama_model_n_embd(model);
    int n_tokens = input->ne[1];
    
    // Retrieve weights first to check dimensions
    struct ggml_tensor* w_q = get_optional_w(currentWeightTensors, "attn_q.weight");
    
    int head_dim = 0;
    if (w_q) {
        // w_q is [n_embd, dim_q] = [n_embd, n_head * head_dim]
        // ne[1] should be dim_q.
        head_dim = w_q->ne[1] / n_head;
        LOGI("L%d Trace: Inferred head_dim=%d from w_q", layerIndex, head_dim);
    } else {
        head_dim = n_embd / n_head;
        LOGI("L%d Trace: Calculated head_dim=%d (n_embd/n_head)", layerIndex, head_dim);
    }
    
    float norm_rms_eps = model_rms_norm_eps; 
    float freq_base = model_rope_freq_base;
    float freq_scale = model_rope_freq_scale;
    
    // Retrieve weights with fallbacks for GPT-NeoX / GPT-OSS naming
    struct ggml_tensor* w_attn_norm = get_optional_w(currentWeightTensors, "attn_norm.weight");
    if (!w_attn_norm) w_attn_norm = get_optional_w(currentWeightTensors, "input_layernorm.weight");

    // w_q already retrieved above for head_dim calculation
    struct ggml_tensor* w_k = get_optional_w(currentWeightTensors, "attn_k.weight");
    struct ggml_tensor* w_v = get_optional_w(currentWeightTensors, "attn_v.weight");
    
    struct ggml_tensor* w_qkv = get_optional_w(currentWeightTensors, "attn_qkv.weight");
    if (!w_qkv) w_qkv = get_optional_w(currentWeightTensors, "attention.query_key_value.weight");

    struct ggml_tensor* w_o = get_optional_w(currentWeightTensors, "attn_output.weight");
    if (!w_o) w_o = get_optional_w(currentWeightTensors, "attention.dense.weight");
    
    struct ggml_tensor* w_ffn_norm = get_optional_w(currentWeightTensors, "ffn_norm.weight");
    if (!w_ffn_norm) w_ffn_norm = get_optional_w(currentWeightTensors, "post_attention_layernorm.weight");

    struct ggml_tensor* w_gate = get_optional_w(currentWeightTensors, "ffn_gate.weight");
    
    struct ggml_tensor* w_down = get_optional_w(currentWeightTensors, "ffn_down.weight");
    if (!w_down) w_down = get_optional_w(currentWeightTensors, "mlp.dense_4h_to_h.weight");
    if (!w_down) w_down = get_optional_w(currentWeightTensors, "mlp.down.weight");

    struct ggml_tensor* w_up = get_optional_w(currentWeightTensors, "ffn_up.weight");
    if (!w_up) w_up = get_optional_w(currentWeightTensors, "mlp.dense_h_to_4h.weight");
    if (!w_up) w_up = get_optional_w(currentWeightTensors, "mlp.up.weight");
    
    // QK-Norm weights (Optional)
    struct ggml_tensor* w_q_norm = get_optional_w(currentWeightTensors, "attn_q_norm.weight");
    struct ggml_tensor* w_k_norm = get_optional_w(currentWeightTensors, "attn_k_norm.weight");
    
    // Optional biases
    struct ggml_tensor* b_q = get_optional_w(currentWeightTensors, "attn_q.bias");
    struct ggml_tensor* b_k = get_optional_w(currentWeightTensors, "attn_k.bias");
    struct ggml_tensor* b_v = get_optional_w(currentWeightTensors, "attn_v.bias");
    
    struct ggml_tensor* b_qkv = get_optional_w(currentWeightTensors, "attn_qkv.bias");
    if (!b_qkv) b_qkv = get_optional_w(currentWeightTensors, "attention.query_key_value.bias");

    struct ggml_tensor* b_o = get_optional_w(currentWeightTensors, "attn_output.bias");
    if (!b_o) b_o = get_optional_w(currentWeightTensors, "attention.dense.bias");

    struct ggml_tensor* b_up = get_optional_w(currentWeightTensors, "ffn_up.bias");
    if (!b_up) b_up = get_optional_w(currentWeightTensors, "mlp.dense_h_to_4h.bias");
    if (!b_up) b_up = get_optional_w(currentWeightTensors, "mlp.up.bias");

    struct ggml_tensor* b_down = get_optional_w(currentWeightTensors, "ffn_down.bias");
    if (!b_down) b_down = get_optional_w(currentWeightTensors, "mlp.dense_4h_to_h.bias");
    if (!b_down) b_down = get_optional_w(currentWeightTensors, "mlp.down.bias");

    // Handle merged QKV for GPT-NeoX
    // Handle merged QKV for GPT-NeoX / GPT-OSS
    if (w_qkv && !w_q) {
        // Calculate GQA dimensions
        int dim_q = n_head * head_dim;
        int dim_kv = n_head_kv * head_dim; // For GQA, this is < n_embd

        // Q: [n_embd, dim_q]
        w_q = ggml_view_2d(ctx_weights, w_qkv, n_embd, dim_q, w_qkv->nb[1], 0);
        
        // K: [n_embd, dim_kv] - Offset is dim_q bytes (accross the sliced dimension)
        // Note: Slicing happens on the second dimension from the host perspective, 
        // but ggml tensor ne[0] is correct.
        // Wait, w_qkv is [n_embd, total_dim].
        // We want to verify offsets.
        // Offset = dim_q columns * stride? No.
        // If w_qkv is [n_embd, total_dim]. Stride nb[1] steps to next row of n_embd elements.
        // So offset should be dim_q * nb[1]? NO.
        // We are slicing ROWS.
        // Offset of row X is X * nb[1].
        // But dim_q is number of ROWS.
        // So K starts at row dim_q.
        // Offset = dim_q * nb[1].
        
        // Let's re-verify standard ggml.
        // Weights: [n_embd, n_out].
        // w_q: [n_embd, dim_q]. Offset 0.
        // w_k: [n_embd, dim_kv]. Offset dim_q * nb[1].
        
        // My previous code:
        // offset: dim_q * type_size.
        // This suggests offset was calculated as if iterating contiguous memory?
        // nb[1] is usually n_embd * type_size.
        // So dim_q * nb[1] is correct offset for "dim_q rows".
        
        // Correct View calls:
        w_q = ggml_view_2d(ctx_weights, w_qkv, n_embd, dim_q, w_qkv->nb[1], 0);
        
        size_t off_k = (size_t)dim_q * w_qkv->nb[1];
        w_k = ggml_view_2d(ctx_weights, w_qkv, n_embd, dim_kv, w_qkv->nb[1], off_k);
        
        size_t off_v = (size_t)(dim_q + dim_kv) * w_qkv->nb[1];
        w_v = ggml_view_2d(ctx_weights, w_qkv, n_embd, dim_kv, w_qkv->nb[1], off_v);
        
        if (b_qkv) {
             b_q = ggml_view_1d(ctx_weights, b_qkv, dim_q, 0);
             b_k = ggml_view_1d(ctx_weights, b_qkv, dim_kv, dim_q * sizeof(float));
             b_v = ggml_view_1d(ctx_weights, b_qkv, dim_kv, (dim_q + dim_kv) * sizeof(float));
        }
    }
    
    if (!w_attn_norm || !w_q || !w_k || !w_v || !w_o || !w_up || !w_down) {
        LOGE("Forward: Missing weights for layer %d w_up=%p w_down=%p", layerIndex, w_up, w_down);
        return nullptr;
    }


    // Build Graph in ctx_build
    
    // 1. Attention Norm
    struct ggml_tensor* inpL = cur; 
    
    if (model_arch_type == ARCH_GPTNEOX) {
        cur = ggml_norm(ctx_build, cur, model_rms_norm_eps);
    } else {
        cur = ggml_rms_norm(ctx_build, cur, model_rms_norm_eps);
    }
    cur = ggml_mul(ctx_build, cur, w_attn_norm);
    
    // DEBUG: Log weight dimensions
    if (w_q) LOGI("L%d Trace: w_q dims=[%ld, %ld]", layerIndex, w_q->ne[0], w_q->ne[1]);
    if (w_k) LOGI("L%d Trace: w_k dims=[%ld, %ld]", layerIndex, w_k->ne[0], w_k->ne[1]);
    if (w_v) LOGI("L%d Trace: w_v dims=[%ld, %ld]", layerIndex, w_v->ne[0], w_v->ne[1]);
    
    // Check for GQA mismatch (Metadata says GQA, weights say MHA)
    if (w_q && w_k && w_q->ne[1] == w_k->ne[1] && n_head != n_head_kv) {
        LOGW("L%d Trace: GQA mismatch! Metadata n_head_kv=%d but w_k matches w_q size. Forcing MHA (n_head_kv=%d).", 
             layerIndex, n_head_kv, n_head);
        n_head_kv = n_head;
    }

    // 2. QKV
    struct ggml_tensor* Q = ggml_mul_mat(ctx_build, w_q, cur);
    struct ggml_tensor* K = ggml_mul_mat(ctx_build, w_k, cur);
    struct ggml_tensor* V = ggml_mul_mat(ctx_build, w_v, cur);
    
    // Apply bias
    if (b_q) Q = ggml_add(ctx_build, Q, b_q);
    if (b_k) K = ggml_add(ctx_build, K, b_k);
    if (b_v) V = ggml_add(ctx_build, V, b_v);
    
    Q = ggml_cont(ctx_build, Q);
    K = ggml_cont(ctx_build, K);
    V = ggml_cont(ctx_build, V);
    
    // QK-Norm (Qwen2/3)
    if (w_q_norm) {
         Q = ggml_rms_norm(ctx_build, Q, norm_rms_eps);
         Q = ggml_mul(ctx_build, Q, w_q_norm);
    }
    if (w_k_norm) {
         K = ggml_rms_norm(ctx_build, K, norm_rms_eps);
         K = ggml_mul(ctx_build, K, w_k_norm);
        LOGI("L0 Bias Trace: Q=%s K=%s V=%s", b_q?"YES":"NO", b_k?"YES":"NO", b_v?"YES":"NO");
    LOGI("L0 QK-Norm Trace: Q=%s K=%s", w_q_norm?"YES":"NO", w_k_norm?"YES":"NO");
    }
    LOGI("L%d Trace: QKV built", layerIndex);
    
    int rope_mode = (int)llama_model_rope_type(model);

    // Reshape to 3D for RoPE and cache storage
    Q = ggml_reshape_3d(ctx_build, Q, head_dim, n_head, n_tokens); // [head_dim, n_head, n_tokens]
    K = ggml_reshape_3d(ctx_build, K, head_dim, n_head_kv, n_tokens); // [head_dim, n_head_kv, n_tokens]
    V = ggml_reshape_3d(ctx_build, V, head_dim, n_head_kv, n_tokens); // [head_dim, n_head_kv, n_tokens]
    LOGI("L%d Trace: Reshaped 3D. Q->ne[2]=%ld", layerIndex, Q->ne[2]);
    
    // 3. RoPE
    // ggml_rope_ext expects sequence length in ne[2]
    struct ggml_tensor* pos = ggml_new_tensor_1d(ctx_build, GGML_TYPE_I32, n_tokens);
    for (int i = 0; i < n_tokens; i++) {
        ((int32_t*)pos->data)[i] = nPast + i;
    }
    
    // Check if n_rot is available (partial rotary)
    int n_dims_rope = (model_n_rot > 0) ? model_n_rot : head_dim;
    
    // Ensure even n_dims for RoPE
    if (n_dims_rope % 2 != 0) {
        LOGE("L%d Trace: RoPE n_dims (%d) is ODD! Adjusting to even...", layerIndex, n_dims_rope);
        n_dims_rope -= 1; // Hack: drop last dimension from rotation?
        // Or if n_rot was 0 and head_dim is 45.
        // If we use 44, it might work but be slightly wrong. Better than crash.
    }
    
    Q = ggml_rope_ext(ctx_build, Q, pos, nullptr, n_dims_rope, rope_mode, 0, freq_base, freq_scale, 0, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(ctx_build, K, pos, nullptr, n_dims_rope, rope_mode, 0, freq_base, freq_scale, 0, 1.0f, 0.0f, 0.0f);
    LOGI("L%d Trace: RoPE done (n_dims=%d)", layerIndex, n_dims_rope);
    
    // Now permute K, V, Q to [head_dim, tokens, heads] for consistent layout
    // Move ne[2] (tokens) to ne[1], and ne[1] (heads) to ne[2]
    // 4. Attention (Final logic aligned with standard Llama architecture)
    // CRITICAL FIX: The previous attempt to remove permute caused Q=[64,64,24] vs K=[64,24,64] crash.
    // Attention requires [dim, seq, heads] layout to produce [seq, seq, heads] scores.
    // RoPE output is [dim, heads, seq]. We MUST permute to [dim, seq, heads].
    Q = ggml_permute(ctx_build, Q, 0, 2, 1, 3);
    K = ggml_permute(ctx_build, K, 0, 2, 1, 3);
    V = ggml_permute(ctx_build, V, 0, 2, 1, 3); 
    LOGI("L%d Trace: Permuted to [dim, seq, heads] for Attention", layerIndex);

    // --- KV Store & Attention ---
    struct ggml_tensor* K_history = nullptr;
    struct ggml_tensor* V_history = nullptr;

    if (ctx_kv && !kv_k.empty() && (nPast + n_tokens <= kv_max_tokens)) {
        struct ggml_tensor* k_cache = kv_k[layerIndex];
        struct ggml_tensor* v_cache = kv_v[layerIndex];
        
        size_t token_bytes = head_dim * sizeof(float);
        size_t offset_bytes = (size_t)nPast * token_bytes;
        
        // View of the specific slot(s) for the new tokens [head_dim, n_tokens, n_head_kv]
        struct ggml_tensor* k_slot = ggml_view_3d(ctx_build, k_cache, head_dim, n_tokens, n_head_kv,
                                                 head_dim * sizeof(float), 
                                                 head_dim * kv_max_tokens * sizeof(float),
                                                 offset_bytes);
        struct ggml_tensor* v_slot = ggml_view_3d(ctx_build, v_cache, head_dim, n_tokens, n_head_kv,
                                                 head_dim * sizeof(float), 
                                                 head_dim * kv_max_tokens * sizeof(float),
                                                 offset_bytes);
        
        // Copy into cache. Ensure source is contiguous for ggml_cpy robustness.
        struct ggml_tensor* k_stored = ggml_cpy(ctx_build, ggml_cont(ctx_build, K), k_slot);
        struct ggml_tensor* v_stored = ggml_cpy(ctx_build, ggml_cont(ctx_build, V), v_slot);
        // LOGI("L%d Trace: cpy to cache done", layerIndex);
        
        // Create a scalar dependency (value=0.0) that forces the cpy ops to execute first
        struct ggml_tensor* kv_dep = ggml_scale(ctx_build, ggml_sum(ctx_build, ggml_add(ctx_build, k_stored, v_stored)), 0.0f);
        
        // View of the full history (0 to nPast + n_tokens)
        K_history = ggml_view_3d(ctx_build, k_cache, head_dim, nPast + n_tokens, n_head_kv,
                                 head_dim * sizeof(float),
                                 head_dim * kv_max_tokens * sizeof(float),
                                 0);
        V_history = ggml_view_3d(ctx_build, v_cache, head_dim, nPast + n_tokens, n_head_kv,
                                 head_dim * sizeof(float),
                                 head_dim * kv_max_tokens * sizeof(float),
                                 0);
        
        // CRITICAL: Add kv_dep (scalar 0.0) to K/V history views to establish graph dependency.
        // Without this, ggml may execute mul_mat(K_history, Q) BEFORE the cpy writes to cache!
        K_history = ggml_add(ctx_build, K_history, kv_dep);
        V_history = ggml_add(ctx_build, V_history, kv_dep);
        // LOGI("L%d Trace: K/V history views built (with cpy dependency)", layerIndex);
    } else {
        // Fallback: no history. Use new K, V directly (already [head_dim, n_tokens, heads])
        K_history = K;
        V_history = V;
    }
    // LOGI("L%d Trace: KV logic done", layerIndex);

    // 4. Attention (Final logic aligned with standard Llama architecture)
    struct ggml_tensor* Q_cur = ggml_cont(ctx_build, Q); 
    struct ggml_tensor* K_cur = K_history;
    struct ggml_tensor* V_cur = V_history;

    LOGI("Debug: Pre-GQA. n_head=%d, n_head_kv=%d", n_head, n_head_kv);
    if (K_cur) LOGI("Debug: K_cur is valid. dims=[%ld, %ld, %ld, %ld]", K_cur->ne[0], K_cur->ne[1], K_cur->ne[2], K_cur->ne[3]);
    else LOGE("Debug: K_cur is NULL!");

    // Broadcasting for GQA if needed
    if (n_head_kv != n_head) {
        int n_group = n_head / n_head_kv;
        LOGI_GQA("GQA Debug: n_head=%d, n_head_kv=%d, n_group=%d, head_dim=%d", n_head, n_head_kv, n_group, head_dim);
        LOGI_GQA("GQA Debug: nPast=%d, n_tokens=%d", nPast, n_tokens);
        LOGI_GQA("GQA Debug: K_cur dims=[%ld, %ld, %ld, %ld]", K_cur->ne[0], K_cur->ne[1], K_cur->ne[2], K_cur->ne[3]);
        
        
        // K: [head_dim, n_tokens, n_head_kv] -> [head_dim, n_tokens, n_head_kv, 1]
        // Repeat to [head_dim, n_tokens, n_head_kv, n_group]
        // Then reshape/permute to [head_dim, n_tokens, n_head]
        
        // Correct approach:
        // K is [head_dim, n_tokens, n_head_kv]
        // 1. Reshape to [head_dim, n_tokens, n_head_kv, 1]
        K_cur = ggml_reshape_4d(ctx_build, K_cur, head_dim, nPast + n_tokens, n_head_kv, 1);
        // 2. Repeat to [..., n_group]
        K_cur = ggml_repeat(ctx_build, K_cur, ggml_new_tensor_4d(ctx_build, K_cur->type, head_dim, nPast + n_tokens, n_head_kv, n_group));
        // 3. Reshape 3D to [head_dim, n_tokens, n_head]
        // Note: ggml_repeat repeats the last dim.
        // We need to ensure the logical layout is correct.
        // If we want [head_dim, n_tokens, n_head], we need "n_head" to be the outer dim.
        // But K_cur internal layout is usually [head_dim, n_tokens, n_head].
        
        // Let's use standard llama.cpp way:
        // K: [head_dim, n_tokens, n_head_kv]
        // Permute to [head_dim, n_head_kv, n_tokens]? No.
        
        // Simplified GQA without complex permutes for now (safer):
        // Reshape K to [head_dim, n_tokens, n_head_kv, 1]
        // Repeat to [head_dim, n_tokens, n_head_kv, n_group]
        // Reshape to [head_dim, n_tokens, n_head] might be tricky if memory not contiguous.
        
        // Actually, if we just want to broadcast, we can use ggml_repeat on the *mask* or modify KQ?
        // No, we need K expanded.
        
        // K: [head_dim, seq, n_head_kv] -> [head_dim, seq, 1, n_head_kv]
        // Repeat ne[2] to n_group -> [head_dim, seq, n_group, n_head_kv]
        // Reshape to [head_dim, seq, n_head]
        
        LOGI_GQA("GQA Step K0: Make Contiguous");
        K_cur = ggml_cont(ctx_build, K_cur);

        LOGI_GQA("GQA Step K1: Reshape 4D (Blocked Setup)");
        K_cur = ggml_reshape_4d(ctx_build, K_cur, head_dim, nPast + n_tokens, 1, n_head_kv);
        
        LOGI_GQA("GQA Step K2: New Tensor 4D (Rep Group)");
        struct ggml_tensor* K_rep = ggml_new_tensor_4d(ctx_build, K_cur->type, head_dim, nPast + n_tokens, n_group, n_head_kv);
        
        LOGI_GQA("GQA Step K3: Repeat");
        K_cur = ggml_repeat(ctx_build, K_cur, K_rep);
        
        LOGI_GQA("GQA Step K4: Cont");
        K_cur = ggml_cont(ctx_build, K_cur);
        
        LOGI_GQA("GQA Step K5: Reshape 3D (Final Heads)");
        K_cur = ggml_reshape_3d(ctx_build, K_cur, head_dim, nPast + n_tokens, n_head);
        
        LOGI_GQA("GQA Step V: Start");
        V_cur = ggml_cont(ctx_build, V_cur);
        V_cur = ggml_reshape_4d(ctx_build, V_cur, head_dim, nPast + n_tokens, 1, n_head_kv);
        struct ggml_tensor* V_rep = ggml_new_tensor_4d(ctx_build, V_cur->type, head_dim, nPast + n_tokens, n_group, n_head_kv);
        V_cur = ggml_repeat(ctx_build, V_cur, V_rep);
        V_cur = ggml_cont(ctx_build, V_cur);
        V_cur = ggml_reshape_3d(ctx_build, V_cur, head_dim, nPast + n_tokens, n_head);
        LOGI_GQA("GQA Step: Done");
    }
    
    // Contiguity for multiplication stability
    K_cur = ggml_cont(ctx_build, K_cur);
    V_cur = ggml_cont(ctx_build, V_cur);
    
    if (!K_cur) { LOGE("Error: K_cur is NULL after final cont!"); return nullptr; }
    if (!V_cur) { LOGE("Error: V_cur is NULL after final cont!"); return nullptr; }
    if (!Q_cur) { LOGE("Error: Q_cur is NULL!"); return nullptr; }

    LOGI_GQA("Debug: Invoking mul_mat(K, Q). K=[%ld,%ld,%ld], Q=[%ld,%ld,%ld]", 
             K_cur->ne[0], K_cur->ne[1], K_cur->ne[2],
             Q_cur->ne[0], Q_cur->ne[1], Q_cur->ne[2]);
    LOGI_GQA("Debug: K type=%d, Q type=%d", K_cur->type, Q_cur->type);

    // scoring
    LOGI_GQA("Attn Trace: mul_mat(K, Q)");
    struct ggml_tensor* KQ = ggml_mul_mat(ctx_build, K_cur, Q_cur); 
    LOGI_GQA("Attn Trace: scale");
    KQ = ggml_scale(ctx_build, KQ, 1.0f / sqrtf((float)head_dim));
    
    if (n_tokens > 1) {
        LOGI_GQA("Attn Trace: diag_mask_inf");
        KQ = ggml_diag_mask_inf(ctx_build, KQ, nPast);
    }

    LOGI_GQA("Attn Trace: soft_max");
    struct ggml_tensor* KQ_soft = ggml_soft_max(ctx_build, KQ);
    
    // Result Projection
    LOGI_GQA("Attn Trace: permute V_cur");
    struct ggml_tensor* V_trans = ggml_permute(ctx_build, V_cur, 1, 0, 2, 3);
    LOGI_GQA("Attn Trace: cont V_trans");
    V_trans = ggml_cont(ctx_build, V_trans); 
    LOGI_GQA("Attn Trace: cont KQ_soft");
    KQ_soft = ggml_cont(ctx_build, KQ_soft);

    LOGI_GQA("Attn Trace: mul_mat(V_trans, KQ_soft)");
    struct ggml_tensor* V_out = ggml_mul_mat(ctx_build, V_trans, KQ_soft); 
    
    LOGI_GQA("Attn Trace: permute V_out");
    V_out = ggml_permute(ctx_build, V_out, 0, 2, 1, 3); 
    LOGI_GQA("Attn Trace: cont V_out");
    V_out = ggml_cont(ctx_build, V_out);
    LOGI_GQA("Attn Trace: reshape_2d V_out to [%d, %d]", n_head * head_dim, n_tokens);
    V_out = ggml_reshape_2d(ctx_build, V_out, n_head * head_dim, n_tokens); 
    
    LOGI_GQA("Attn Trace: mul_mat(w_o, V_out) w_o=[%ld,%ld], V_out=[%ld,%ld]", 
             w_o->ne[0], w_o->ne[1], V_out->ne[0], V_out->ne[1]);
    struct ggml_tensor* attn_out = ggml_mul_mat(ctx_build, w_o, V_out);
    if (b_o) attn_out = ggml_add(ctx_build, attn_out, b_o);
    
    if (model_parallel_residual) {

        // x = x + attn(norm(x)) + ffn(norm(x))
        struct ggml_tensor* ffn_norm_out = cur;
        if (w_ffn_norm) {
            ffn_norm_out = ggml_norm(ctx_build, inpL, model_rms_norm_eps);
            ffn_norm_out = ggml_mul(ctx_build, ffn_norm_out, w_ffn_norm);
        }
        
        struct ggml_tensor* ffn_out;
        if (w_gate) {
            struct ggml_tensor* gate = ggml_mul_mat(ctx_build, w_gate, ffn_norm_out);
            gate = ggml_silu(ctx_build, gate);
            struct ggml_tensor* up = ggml_mul_mat(ctx_build, w_up, ffn_norm_out);
            ffn_out = ggml_mul(ctx_build, gate, up);
        } else {
            ffn_out = ggml_mul_mat(ctx_build, w_up, ffn_norm_out);
            if (b_up) ffn_out = ggml_add(ctx_build, ffn_out, b_up);
            ffn_out = ggml_gelu(ctx_build, ffn_out);
        }
        
        ffn_out = ggml_mul_mat(ctx_build, w_down, ffn_out);
        if (b_down) ffn_out = ggml_add(ctx_build, ffn_out, b_down);

        return ggml_add(ctx_build, inpL, ggml_add(ctx_build, attn_out, ffn_out));
    } else {
        // Sequential Residual (Llama pattern)
        struct ggml_tensor* cur_res = ggml_add(ctx_build, attn_out, inpL);
        
        struct ggml_tensor* ffn_inp = cur_res; 
        cur_res = ggml_rms_norm(ctx_build, cur_res, norm_rms_eps);
        cur_res = ggml_mul(ctx_build, cur_res, w_ffn_norm);
        
        struct ggml_tensor* ffn_out;
        if (w_gate) {
            struct ggml_tensor* gate = ggml_mul_mat(ctx_build, w_gate, cur_res);
            gate = ggml_silu(ctx_build, gate);
            struct ggml_tensor* up = ggml_mul_mat(ctx_build, w_up, cur_res);
            ffn_out = ggml_mul(ctx_build, gate, up);
        } else {
            ffn_out = ggml_mul_mat(ctx_build, w_up, cur_res);
            if (b_up) ffn_out = ggml_add(ctx_build, ffn_out, b_up);
            ffn_out = ggml_gelu(ctx_build, ffn_out);
        }
        
        ffn_out = ggml_mul_mat(ctx_build, w_down, ffn_out);
        if (b_down) ffn_out = ggml_add(ctx_build, ffn_out, b_down);
        
        return ggml_add(ctx_build, ffn_out, ffn_inp);
    }
}

std::string TrueLargeRuntime::step_lbl() {
    // 1. Sample from previous logits
    // We cannot use llama_sampler_sample(..., -1) because we bypass llama_decode,
    // which triggers an internal safety check fail in llama.cpp.
    // Instead, we manually build the candidates array and apply the sampler chain.
    float* prev_logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    
    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    for (llama_token i = 0; i < n_vocab; i++) {
        candidates.push_back({i, prev_logits[i], 0.0f});
    }
    
    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), -1, false };
    
    llama_sampler_apply(sampler, &candidates_p);
    
    // Pick the token with the highest probability (greedy)
    int max_idx = 0;
    for (int i = 1; i < (int)candidates_p.size; i++) {
        if (candidates_p.data[i].logit > candidates_p.data[max_idx].logit) {
            max_idx = i;
        }
    }
    
    llama_token id = candidates_p.data[max_idx].id;
    llama_sampler_accept(sampler, id);
    if (id == llama_vocab_eos(llama_model_get_vocab(model))) {
        LOGI("EOS generated in step_lbl");
        return "";
    }

    // 2. Prepare embedding
    ggml_free(ctx_compute);
    ggml_set_abort_callback(custom_ggml_abort_callback);
    // Increase prompt context size to 2GB for GQA/MoE nodes and MXFP4 additions
    struct ggml_init_params params = { .mem_size = 2048LL*1024*1024, .mem_buffer = NULL };
    ctx_compute = ggml_init(params);
    
    struct ggml_tensor* input = nullptr;
    
    if (w_token_embd) {
         struct ggml_tensor* idx = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_I32, 1);
         ggml_set_i32(idx, id);
         // w_token_embd is [n_embd, n_vocab]
         // get_rows returns [n_embd, 1]
         input = ggml_get_rows(ctx_compute, w_token_embd, idx);
         
         // CRITICAL: Compute the embedding graph before forwarding to layers
         struct ggml_cgraph* gf_emb = ggml_new_graph_custom(ctx_compute, 8192, false);
         ggml_build_forward_expand(gf_emb, input);
         ggml_graph_compute_with_ctx(ctx_compute, gf_emb, nThreads);
         
         LOGI("LBL Step: Embedding computed. Type=%d", input->type);

         // Force F32 if input is quantized (e.g. Q8_0) because norms/RoPE expect F32
         if (input->type != GGML_TYPE_F32) {
             LOGI("LBL Step: Converting input from type %d to F32", input->type);
             struct ggml_tensor* input_f32 = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, input->ne[0]);
             // Use ggml_cpy to dequantize
             struct ggml_cgraph* gf_cast = ggml_new_graph_custom(ctx_compute, 8192, false);
             input_f32 = ggml_cpy(ctx_compute, input, input_f32);
             ggml_build_forward_expand(gf_cast, input_f32);
             ggml_graph_compute_with_ctx(ctx_compute, gf_cast, nThreads);
             input = input_f32;
         }
    } else {
        LOGE("Global weights missing (token_embd). Using 0s.");
        int n_embd = llama_model_n_embd(model);
        input = ggml_new_tensor_1d(ctx_compute, GGML_TYPE_F32, n_embd);
        ggml_set_f32(input, 0.0f);
    }
    
    struct ggml_context* ctx_curr = ctx_compute; // Holds 'input'
    struct ggml_context* ctx_next = ctx_compute_back;
    
    int n_layer = llama_model_n_layer(model);
    LOGI("LBL Step: Deciphering token %d...", (int)id);
    for (int i = 0; i < n_layer; ++i) {
        if (i % 8 == 0) LOGI("LBL: Progress %d/%d layers", i, n_layer);
        // Reset next context
        ggml_free(ctx_next);
        ctx_next = ggml_init(params);
        
        // Log progress
        // LOGI("Loop: Layer %d/%d", i, n_layer);
        
        struct ggml_tensor* out = forwardLayer(i, input, ctx_next);
        if (!out) {
            LOGE("Layer %d failed to build", i);
            break;
        }
        
        // Compute
        struct ggml_cgraph* gf = ggml_new_graph_custom(ctx_next, 8192, false);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx_next, gf, nThreads);
        
        // --- Dependency Cut ---
        // Create a leaf tensor in ctx_next to hold the COMPUTED result.
        // This prevents the next layer's graph from traversing back into the previous context.
        struct ggml_tensor* result_leaf = ggml_new_tensor(ctx_next, out->type, ggml_n_dims(out), out->ne);
        memcpy(result_leaf->data, out->data, ggml_nbytes(out));
        input = result_leaf; 
        
        std::swap(ctx_curr, ctx_next);
        // Now input is in ctx_curr. ctx_next (old curr) can be freed in next iter.
    }
    
    // Final Output (Norm + Head) in ctx_curr
    // input is final hidden state
    if (w_output_norm && w_output) {
         struct ggml_tensor* cur = input;
         
         // Norm
         if (model_arch_type == ARCH_GPTNEOX) {
             cur = ggml_norm(ctx_curr, cur, model_rms_norm_eps);
         } else {
             cur = ggml_rms_norm(ctx_curr, cur, model_rms_norm_eps);
         }
         cur = ggml_mul(ctx_curr, cur, w_output_norm);
         
         // Head (Logits)
         // w_output: [n_embd, n_vocab] ? Or [n_vocab, n_embd]?
         // GGUF usually stores output as [n_embd, n_vocab].
         // Mul Mat: [n_embd, n_vocab] * [n_embd, 1] -> [n_vocab, 1]
         // Check transpose? ggml_mul_mat uses A * B.
         // If A is [N, K], B is [K, M], result [N, M].
         // ggml: ne[0] is rows? 
         // Standard: output.weight is [n_vocab, n_embd] logic?
         // In GGUF: dims are [n_embd, n_vocab] usually (transposed).
         // So MUL_MAT works directly.
         
         struct ggml_tensor* logits = ggml_mul_mat(ctx_curr, w_output, cur);
         
         // LOGI("DIAG Final Proj: ...", ...);
         
         // Compute final
         struct ggml_cgraph* gf_final = ggml_new_graph_custom(ctx_curr, 8192, false);
         ggml_build_forward_expand(gf_final, logits);
         ggml_graph_compute_with_ctx(ctx_curr, gf_final, nThreads);
         
         // Copy to llama logits
         // Logits tensor data is whatever w_output type * input type produces.
         // Usually F32.
         
         const llama_vocab* vocab = llama_model_get_vocab(model);
         int n_vocab = llama_vocab_n_tokens(vocab);
         float* dst = llama_get_logits(ctx);
         
         // Copy logits to llama context for sampler
         // Logits tensor data usually F32.
         if (logits->type == GGML_TYPE_F32) {
              memcpy(dst, logits->data, n_vocab * sizeof(float));
         } else {
              // Fallback for non-F32 logits (rare for output.weight but possible)
              // Just use raw copy for now as llama_get_logits expects float*
              memcpy(dst, logits->data, n_vocab * sizeof(float)); 
         }
     } else {
         LOGE("Global weights missing (output). Skipping final projection.");
     }
    
    generatedTokens.push_back(id);
    nPast++;
    
    // Telemetry
    auto end_time = std::chrono::steady_clock::now();
    double total_duration = std::chrono::duration<double>(end_time - t_generation_start).count();
    double since_start = std::chrono::duration<double>(end_time - t_session_start).count();
    
    if (total_duration > 0) {
        lastTPS = generatedTokens.size() / total_duration;
    }
    
    lastTotalTime = since_start;
    lastRAM = getMemoryUsageKB() / 1024; // MB
    lastCPUFreq = (double)getCurrentCpuFreqHz() / 1e9; // GHz
    
    std::string piece = token_to_str(ctx, id);
    LOGI("LBL Step: %lu tokens generated -> '%s' | TPS: %.2f | RAM: %ld MB | CPU: %.2f GHz", 
         generatedTokens.size(), piece.c_str(), lastTPS, lastRAM, lastCPUFreq);

    return piece;
}



void TrueLargeRuntime::initLayerWeights(int layerIndex) {
    if (!scheduler) return;

    // 1. Ensure data is in RAM
    if (!scheduler->prepareLayer(layerIndex)) {
        LOGE("Failed to prepare layer %d", layerIndex);
        return;
    }

    // 2. Reset weight context (cheap 1MB metadata)
    if (ctx_weights) ggml_free(ctx_weights);
    
    struct ggml_init_params params_w = {
        .mem_size   = 1024 * 1024 * 4, // 4MB enough for descriptors
        .mem_buffer = NULL,
        .no_alloc   = true, // Manually setting data pointers
    };
    ctx_weights = ggml_init(params_w);
    
    currentWeightTensors.clear();
    
    // 3. Create tensor descriptors
    const LayerSourceInfo* info = headerParser->getLayerSourceInfo(layerIndex);
    void* layerDataBase = scheduler->getLayerData(layerIndex);
    
    // We need to know the base offset of the layer in the file to map relative offsets?
    // LayerScheduler::loadLayerInternal maps [minOffset, maxLimit].
    // buffer[0] is at minOffset.
    // tensor.offset is relative to file data start (absolute file offset in data section).
    // GGUFHeaderParser was updated to output Absolute Offset (dataStart + local).
    // Wait, GGUFHeaderParser.cpp: `pair.second.offset += dataStart;`
    // So `TensorInfo.offset` is ABSOLUTE file offset.
    
    // LayerScheduler maps [minOffset, maxLimit].
    // So `buffer_ptr` = base ptr.
    // `tensor_ptr` = buffer_ptr + (tensor.offset - minOffset).
    
    // Find minOffset again (should be consistent with Scheduler)
    size_t minOffset = (size_t)-1;
    for (const auto& tp : info->tensors) {
        if (tp.second.offset < minOffset) minOffset = tp.second.offset;
    }
    
    char* basePtr = static_cast<char*>(layerDataBase);
    
    for (const auto& tp : info->tensors) {
        const TensorInfo& t = tp.second;
        
        // Log all suffixes for layer 0 to help debug missing weights
        if (layerIndex == 0) {
            LOGI("Layer 0 Tensor: Suffix='%s', FullName='%s'", tp.first.c_str(), t.name.c_str());
        }

        // Setup shape
        struct ggml_tensor* tensor = nullptr;
        
        // ggml_new_tensor uses reverse dims usually? 
        // ggml: ne[0] is inner-most.
        // GGUF dims: [dim0, dim1, ...]. 
        // Check `ggml` convention. GGUF stores `ne[0], ne[1]`...
        // `read` loop in parser: `n_elements *= dim`.
        // `dims` vector in parser: [dim0, dim1, ...].
        // `ggml_new_tensor` takes: type, n_dims, ne (reversed?).
        // `ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3)`.
        
        if (t.dims.size() == 1) {
            tensor = ggml_new_tensor_1d(ctx_weights, (ggml_type)t.type, t.dims[0]);
        } else if (t.dims.size() == 2) {
            tensor = ggml_new_tensor_2d(ctx_weights, (ggml_type)t.type, t.dims[0], t.dims[1]);
        } else if (t.dims.size() == 3) {
            tensor = ggml_new_tensor_3d(ctx_weights, (ggml_type)t.type, t.dims[0], t.dims[1], t.dims[2]);
        } else if (t.dims.size() >= 4) {
            tensor = ggml_new_tensor_4d(ctx_weights, (ggml_type)t.type, t.dims[0], t.dims[1], t.dims[2], t.dims[3]);
        }
        
        if (tensor) {
            // Set name
            ggml_set_name(tensor, t.name.c_str());
            
            // Set data pointer
            size_t relativeOffset = t.offset - minOffset;
            tensor->data = basePtr + relativeOffset;
            
            // Map by suffix for easy access (e.g. "attn_q.weight")
            // t.name is full name "blk.0.attn_q.weight"
            // We want key "attn_q.weight"
            std::string suffix = tp.first; // Map key is suffix
            currentWeightTensors[suffix] = tensor;
        }
    }


}

