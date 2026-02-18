#include "LayerScheduler.h"
#include <iostream>
#include <algorithm>
#include <android/log.h>
#include <sys/mman.h>

#define TAG "TrueLargeLBL"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

LayerScheduler::LayerScheduler(const std::string& path, const GGUFHeaderParser* parserPtr, int maxMem)
    : modelPath(path), parser(parserPtr), maxLayersInMemory(maxMem), loader(path), 
      prefetchRunning(false), nextPrefetchLayer(-1) {
    if (!loader.init()) {
        LOGE("LayerScheduler: Failed to initialize loader for %s", path.c_str());
    }
}

LayerScheduler::~LayerScheduler() {
    stopPrefetcher();
    // Unique pointers clean up themselves
    weightBuffers.clear();
}

bool LayerScheduler::prepareLayer(int layerIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    
    if (loadedLayers.count(layerIndex)) {
        return true; // Already loaded
    }

    // Check memory pressure / max layers
    if (loadedLayers.size() >= maxLayersInMemory) {
        // Simple eviction: remove oldest (lowest index)
        // Ideally we remove 'furthest from current execution'
        // If we are at layer N, we need N. We probably don't need N-2.
        // Let's iterate and find best candidate.
        // Assuming forward pass 0->N, we evict smallest index?
        // But what if we are at N and prefetching N+1?
        // We want to keep N, N+1. Evict N-1.
        
        int evictCandidate = -1;
        for (int loaded : loadedLayers) {
            // Heuristic: evict absolute furthest? Or just smallest if sequential?
            // Sequential assumption: evict smallest index that is < layerIndex
            if (loaded < layerIndex) {
                 if (evictCandidate == -1 || loaded < evictCandidate) {
                     evictCandidate = loaded;
                 }
            }
        }
        
        if (evictCandidate != -1) {
            // Internal release (lock already held)
            void* data = weightBuffers[evictCandidate]->getData();
            size_t size = weightBuffers[evictCandidate]->getSize();
            if (data) {
                madvise(data, size, MADV_DONTNEED);
            }
            weightBuffers.erase(evictCandidate);
            loadedLayers.erase(evictCandidate);
            LOGI("Evicted layer %d (Hinted OS to reclaim RAM)", evictCandidate);
        } else {
             // If all loaded layers are > layerIndex (unlikely in forward pass),
             // or maxLayers is too small.
             if (loadedLayers.size() >= maxLayersInMemory) {
                 // Force evict ANY layer that isn't current?
                 int first = *loadedLayers.begin();
                 void* data = weightBuffers[first]->getData();
                 size_t size = weightBuffers[first]->getSize();
                 if (data) {
                     madvise(data, size, MADV_DONTNEED);
                 }
                 weightBuffers.erase(first);
                 loadedLayers.erase(first);
                 LOGI("Fallback Evicted layer %d (Hinted OS)", first);
             }
        }
    }

    return loadLayerInternal(layerIndex);
}

bool LayerScheduler::loadLayerInternal(int layerIndex) {
    const LayerSourceInfo* info = parser->getLayerSourceInfo(layerIndex);
    if (!info) {
        LOGE("LayerScheduler: No info for layer %d", layerIndex);
        return false;
    }
    
    // 1. Calculate total size
    // We want to pack tensors tightly.
    // But offsets in file might have gaps (alignment).
    // If we pack tightly in RAM, we must track the new relative offsets for each tensor.
    // HOWEVER, GGUFHeaderParser::TensorInfo stores 'offset' relative to data start.
    // If we want to use 'offset' as is, we need to map the whole range [min_offset, max_offset + size].
    // Then we just use (buffer + (tensor.offset - min_offset)).
    // This wastes RAM if there are huge gaps, but GGUF usually aligns to 32 bytes, so gaps are tiny.
    
    // Find bounds
    size_t minOffset = (size_t)-1;
    size_t maxLimit = 0;
    
    for (const auto& tp : info->tensors) {
        if (tp.second.offset < minOffset) minOffset = tp.second.offset;
        size_t end = tp.second.offset + tp.second.size;
        if (end > maxLimit) maxLimit = end;
    }
    
    if (minOffset == (size_t)-1) return false; // Empty layer?
    
    size_t totalRange = maxLimit - minOffset;
    
    // Zero-Copy Optimization:
    // Instead of malloc + memcpy, we adopt the mmap directly.
    LayerMap lm = loader.loadLayerMap(minOffset, totalRange);
    if (!lm.data) {
        LOGE("LayerScheduler: Failed to map layer %d", layerIndex);
        return false;
    }
    
    auto buffer = std::make_unique<WeightBuffer>();
    buffer->adoptMmap(lm.data, totalRange, lm.fullMapPtr, lm.fullMapSize);
    
    // Store
    weightBuffers[layerIndex] = std::move(buffer);
    loadedLayers.insert(layerIndex);
    
    LOGI("Loaded layer %d (Zero-Copy Mmap: %zu bytes)", layerIndex, totalRange);
    return true;
}

void LayerScheduler::releaseLayer(int layerIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    if (weightBuffers.count(layerIndex)) {
        weightBuffers.erase(layerIndex);
        loadedLayers.erase(layerIndex);
    }
}

void* LayerScheduler::getLayerData(int layerIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = weightBuffers.find(layerIndex);
    if (it != weightBuffers.end()) {
        return it->second->getData();
    }
    return nullptr;
}

size_t LayerScheduler::getLayerSize(int layerIndex) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = weightBuffers.find(layerIndex);
    if (it != weightBuffers.end()) {
        return it->second->getSize();
    }
    return 0;
}

void LayerScheduler::startPrefetcher() {
    // To be implemented in Phase 3
}

void LayerScheduler::stopPrefetcher() {
    // To be implemented in Phase 3
}

void LayerScheduler::queuePrefetch(int layerIndex) {
    // To be implemented in Phase 3
}
