#ifndef LAYER_WEIGHTS_H
#define LAYER_WEIGHTS_H

#include "ggml.h"
#include <cstddef>

/**
 * Holds pointers to the ggml_tensors for a single transformer layer.
 * These tensors point to data managed by the LayerScheduler/WeightBuffer.
 */
struct LayerWeights {
    // Attention mechanism weights
    struct ggml_tensor* attn_q = nullptr;
    struct ggml_tensor* attn_k = nullptr;
    struct ggml_tensor* attn_v = nullptr;
    struct ggml_tensor* attn_output = nullptr;
    
    // Feed-forward network weights
    struct ggml_tensor* ffn_gate = nullptr; // w1
    struct ggml_tensor* ffn_down = nullptr; // w2
    struct ggml_tensor* ffn_up = nullptr;   // w3
    
    // Normalization weights
    struct ggml_tensor* attn_norm = nullptr;
    struct ggml_tensor* ffn_norm = nullptr;
    
    // Metadata
    size_t totalSize = 0;   // Total size in bytes of all weights in this layer
    size_t fileOffset = 0;  // Offset in the GGUF file where this layer's data begins
    int layerIndex = -1;
};

#endif // LAYER_WEIGHTS_H
