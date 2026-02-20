#ifndef LAYER_SCHEDULER_H
#define LAYER_SCHEDULER_H

#include <map>
#include <deque>
#include <set>
#include <vector>
#include <memory>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include "LayerLoader.h"
#include "GGUFHeaderParser.h"
#include "WeightBuffer.h"

/**
 * Manages layer memory loading and unloading.
 * Ensures that only the necessary layers are in RAM.
 */
class LayerScheduler {
public:
    LayerScheduler(const std::string& modelPath, const GGUFHeaderParser* parser, int maxLayersInMemory = 3);
    ~LayerScheduler();

    // Prepare a layer for computation (load if not loaded)
    bool prepareLayer(int layerIndex);

    // Release a layer from memory
    void releaseLayer(int layerIndex);

    // Get the base pointer to the weights for a layer
    // Returns nullptr if not loaded
    void* getLayerData(int layerIndex);

    // Get size of layer data
    size_t getLayerSize(int layerIndex);
    
    int getMaxLayersInMemory() const { return maxLayersInMemory; }

    // Start background prefetcher (optional)
    void startPrefetcher();
    void stopPrefetcher();
    void queuePrefetch(int layerIndex);

private:
    std::string modelPath;
    const GGUFHeaderParser* parser;
    int maxLayersInMemory;
    
    // Layer Loader instance
    LayerLoader loader;
    
    // Manages buffers: Layer Index -> Buffer
    std::map<int, std::unique_ptr<WeightBuffer>> weightBuffers;
    
    // Track loaded layers
    std::set<int> loadedLayers;
    
    // Mutex for thread safety
    std::mutex mutex;
    
    // TRACKING: Currently used in compute
    std::atomic<int> activeComputeLayer;
    
    // Prefetch logic
    std::atomic<bool> prefetchRunning;
    std::thread prefetchThread;
    // Simple queue logic (or just a target)
    std::deque<int> prefetchQueue;
    int currentPrefetchingLayer; // -1 if idle
    
    // Sync for prefetcher
    std::mutex prefetchMutex;
    std::condition_variable prefetchCv;
    bool prefetchReady; // Dummy flag if needed, usually atomic nextPrefetch suffices
    
    void prefetchThreadLoop();
    
    // Helper to actually load
    bool loadLayerInternal(int layerIndex);
};

#endif // LAYER_SCHEDULER_H
