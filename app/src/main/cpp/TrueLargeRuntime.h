#ifndef TRUELARGE_RUNTIME_H
#define TRUELARGE_RUNTIME_H

#include <string>
#include <memory>
#include <functional>
#include <vector>

#include "GGUFHeaderParser.h"
#include "LayerLoader.h"
#include "llama.h" 

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
    bool createSession(const std::string& prompt);

    // 4. Generate next token (step)
    // Returns the token string piece, or empty string/special value for EOS/Error
    std::string step();

    // 5. Context window info
    int getContextTrain();
    int getContextCurrent();

    // 6. Release resources
    void release();

private:
    std::string modelPath;
    int nThreads = 4;
    int nGpuLayers = 0;
    
    // Sampler params
    float temperature = 0.7f;
    int topK = 40;
    float topP = 0.9f;

    std::unique_ptr<GGUFHeaderParser> headerParser;
    std::unique_ptr<LayerLoader> layerLoader;

    // llama.cpp structures
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    struct llama_sampler* sampler = nullptr;
    
    // Internal state for generation
    std::vector<llama_token> generatedTokens;
    int nPast = 0;
    std::chrono::steady_clock::time_point t_session_start;
    std::chrono::steady_clock::time_point t_generation_start;

    // Helper: The core logic to run one layer manually
    void computeLayer(int layerIndex);
};

#endif // TRUELARGE_RUNTIME_H
