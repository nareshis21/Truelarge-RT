#ifndef TRUELARGE_RUNTIME_H
#define TRUELARGE_RUNTIME_H

#include <string>
#include <memory>
#include <functional>
#include <vector>

#include "GGUFHeaderParser.h"
#include "LayerLoader.h"
#include "LayerScheduler.h"
#include "llama.h" 
#include <mutex>

class TrueLargeRuntime {
public:
    TrueLargeRuntime();
    ~TrueLargeRuntime();

    // 1. Initialize the engine with model path
    bool loadModel(const std::string& modelPath);

    // 2. Configure inference parameters
    void configure(int threads, int gpuLayers);
    void configureSampler(float temp, int k, float p);

    // 3. Create context for a prompt
    bool createSession(const std::string& prompt, bool keepHistory = false);

    // 4. Generate next token (step)
    // Returns the token string piece, or empty string/special value for EOS/Error
    std::string step();

    // 5. Context window info
    int getContextTrain();
    int getContextCurrent();

    // 6. Release resources
    void release();

    // Benchmark Telemetry (Public for JNI access)
    double lastTTFT = 0.0;
    double lastTPS = 0.0;
    long lastRAM = 0;
    double lastCPUFreq = 0.0;
    double lastTotalTime = 0.0;
    std::string getInferenceMode() const { return inferenceMode; }

private:
    std::string modelPath;
    std::string inferenceMode = "Unknown";
    int nThreads = 4;
    int nGpuLayers = 0;
    
    // Sampler params
    float temperature = 0.7f;
    int topK = 40;
    float topP = 0.9f;

    std::unique_ptr<GGUFHeaderParser> headerParser;
    std::unique_ptr<LayerLoader> layerLoader;

    // Layer-by-Layer Components
    std::unique_ptr<LayerScheduler> scheduler;
    bool useLayerByLayer = false;
    
    // Custom Compute State (ggml)
    struct ggml_context* ctx_compute = nullptr;      // Ping
    struct ggml_context* ctx_compute_back = nullptr; // Pong
    struct ggml_context* ctx_weights = nullptr;      // For weight tensors
    struct ggml_context* ctx_global = nullptr;       // For embeddings/output
    
    // KV Cache for LBL
    struct ggml_context* ctx_kv = nullptr;
    std::vector<struct ggml_tensor*> kv_k;
    std::vector<struct ggml_tensor*> kv_v;
    int kv_max_tokens = 512; 
    
    // Global weights
    struct ggml_tensor* w_token_embd = nullptr;
    struct ggml_tensor* w_output_norm = nullptr;
    struct ggml_tensor* w_output = nullptr;
    
    // Current hidden state
    struct ggml_tensor* cur_hidden_state = nullptr;
    struct ggml_tensor* cur_embeddings = nullptr; 

    // llama.cpp structures (Target Model)
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    struct llama_sampler* sampler = nullptr;

    // Speculative Decoding (Draft Model)
    llama_model* model_dft = nullptr;
    llama_context* ctx_dft = nullptr;
    struct llama_sampler* sampler_dft = nullptr;
    int nPastDft = 0;
    std::vector<llama_token> speculativeBuffer;

    // Internal state for generation
    std::vector<llama_token> generatedTokens;
    int nPast = 0;
    std::chrono::steady_clock::time_point t_generation_start;
    std::chrono::steady_clock::time_point t_session_start;
    
    enum ArchType {
        ARCH_LLAMA,
        ARCH_GPTNEOX,
        ARCH_QWEN,
        ARCH_GEMMA,
        ARCH_UNKNOWN
    };

    ArchType model_arch_type = ARCH_LLAMA;

    // Model HParams for LBL
    float model_rope_freq_base = 10000.0f;
    float model_rope_freq_scale = 1.0f;
    float model_rms_norm_eps = 1e-5f;
    int model_n_rot = 0;
    bool model_parallel_residual = false;


    // Map for current layer weights
    std::map<std::string, struct ggml_tensor*> currentWeightTensors;

    // Helper: The core logic to run one layer manually
    // Returns the output tensor of the layer. Builds graph in 'ctx_build'.
    struct ggml_tensor* forwardLayer(int layerIndex, struct ggml_tensor* input, struct ggml_context* ctx_build);
    
    // Initialize standard llama weights wrappers for a layer
    void initLayerWeights(int layerIndex);
    
    // Load global weights (embeddings, output_norm, output)
    void initGlobalWeights();
    
    // Step functionality for LBL
    std::string step_lbl();
    
    // Check if we should use layer-by-layer loading
    bool detectLayerByLayerNeeded(long fileSizeKB);
    
    // Full init for LBL mode
    void initLayerByLayer();
    
    // Metadata helper
    float getModelMetaFloat(const char* key, float defaultValue);

    std::recursive_mutex engine_mutex;
};


#endif // TRUELARGE_RUNTIME_H
