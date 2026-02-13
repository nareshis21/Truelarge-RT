#include "TrueLargeRuntime.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <android/log.h>
#include <cstring> // For memset, memcpy if needed

#define TAG "TrueLargePerf"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

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

// Helper to set CPU affinity (prefer big cores)
void set_cpu_affinity() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    int n_cores = sysconf(_SC_NPROCESSORS_CONF);
    // Prefer the last 4 cores (usually Big cores like Cortex-X or A7xx on Android)
    // Most octa-core chips have 4 big cores at the higher indices.
    for (int i = std::max(0, n_cores - 4); i < n_cores; i++) {
        CPU_SET(i, &cpuset);
    }
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        LOGE("Failed to set CPU affinity");
    }
}

TrueLargeRuntime::TrueLargeRuntime() {}

TrueLargeRuntime::~TrueLargeRuntime() {
    release();
}

bool TrueLargeRuntime::loadModel(const std::string& path) {
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
    
    if (availRamKB > (fileSizeKB + 1024 * 1024)) {
        LOGI("Sufficient RAM detected (Avail: %ld KB, Model: %ld KB). Locking model in memory.", availRamKB, fileSizeKB);
        model_params.use_mlock = true;
    } else {
        LOGI("Low RAM detected (Avail: %ld KB, Model: %ld KB). Using mmap (paging).", availRamKB, fileSizeKB);
        model_params.use_mlock = false; // Default is usually false/true depending on ver, but explicit here
    }

    // API change: llama_load_model_from_file -> llama_model_load_from_file
    model = llama_model_load_from_file(path.c_str(), model_params);
    if (!model) {
        LOGE("Failed to load model: %s", path.c_str());
        return false;
    }

    // Cap threads for mobile (usually 4 is sweet spot)
    if (nThreads > 4) nThreads = 4;
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
    set_cpu_affinity(); // Pin to big cores for this session
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
    
    // Add samplers to chain
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234)); // Fixed seed for now, or use LLAMA_DEFAULT_SEED

    // Tokenize
    std::vector<llama_token> tokens;
    // Over-allocate slightly to be safe
    tokens.resize(prompt.size() + 10); 
    
    // API change: needs vocab
    const llama_vocab* vocab = llama_model_get_vocab(model);
    
    // When keeping history, usually we don't want to re-add BOS if nPast > 0
    bool add_bos = (nPast == 0);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), add_bos, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), tokens.size(), add_bos, false);
    }
    tokens.resize(n_tokens);

    LOGI("Tokenized prompt: %d tokens. History: %d. \"%s\"", n_tokens, nPast, prompt.c_str());

    if (!keepHistory) {
        // Clear KV cache for new session
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_seq_rm(mem, -1, 0, -1);
        
        nPast = 0;
        generatedTokens.clear();
    }

    // Prepare batch manually since helper is missing

    // Prepare batch manually since helper is missing
    if (g_batch_init) {
        llama_batch_free(g_batch);
    }
    // Max batch size should be enough for the prompt
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

    // Last token needs to output logits
    g_batch.logits[g_batch.n_tokens - 1] = true;

    // Decode prompt
    auto start = std::chrono::high_resolution_clock::now();
    
    if (llama_decode(ctx, g_batch) != 0) {
        LOGE("llama_decode failed");
        return false;
    }

    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    nPast += g_batch.n_tokens;
    t_generation_start = std::chrono::steady_clock::now();
    LOGI("Prompt Eval Speed: %.2f ms for %d tokens (%.2f t/s)", duration, n_tokens, (n_tokens / duration) * 1000.0);
    
    return true;
}

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

std::string TrueLargeRuntime::step() {
    if (!ctx || !sampler) return "";

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
        auto ttft = std::chrono::duration<double, std::milli>(end - t_session_start).count();
        LOGI("TTFT: %.2f ms", ttft);
    }

    double tps = generatedTokens.size() / std::chrono::duration<double>(end - t_generation_start).count();
    
    // Telemetry: RAM and CPU
    long rss_kb = getMemoryUsageKB();
    long avail_kb = getAvailableMemoryKB();
    long freq_hz = getCurrentCpuFreqHz();
    int cpu_id = sched_getcpu();
    
    const char* warning = "";
    if (avail_kb < 512 * 1024) { // Warning if less than 512MB free
        warning = "[LOW-RAM IO-WAIT] ";
    }

    LOGI("%sGen: %d -> Token %d ('%s') | Speed: %.2f ms | TPS: %.2f | RAM: %ld MB | CPU: #%d @ %.2f GHz", 
         warning, (int)generatedTokens.size(), next_token, piece.c_str(), duration, tps, rss_kb / 1024, cpu_id, freq_hz / 1e9);

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
    llama_backend_free();
}

void TrueLargeRuntime::computeLayer(int layerIndex) {
    // Placeholder
}
