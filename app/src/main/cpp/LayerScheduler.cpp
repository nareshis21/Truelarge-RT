#include "LayerScheduler.h"
#include <iostream>
#include <algorithm>
#include <android/log.h>
#include <sys/mman.h>

#define TAG "TrueLargeLBL"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

LayerScheduler::LayerScheduler(const std::string& path, const GGUFHeaderParser* parserPtr, int maxMem)
    : modelPath(path), parser(parserPtr), maxLayersInMemory(maxMem), loader(path), 
      activeComputeLayer(-1), prefetchRunning(false), currentPrefetchingLayer(-1) {
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
    // Mark as active immediately so prefetcher doesn't evict it
    activeComputeLayer = layerIndex;

    std::unique_lock<std::mutex> lock(mutex);

    if (loadedLayers.count(layerIndex)) {
        return true; // Already loaded
    }

    // NEW: If this layer is currently being prefetched or in queue, wait for it
    {
        std::unique_lock<std::mutex> pfLock(prefetchMutex);
        bool inQueueOrLoading = (currentPrefetchingLayer == layerIndex);
        if (!inQueueOrLoading) {
            for (int q : prefetchQueue) {
                if (q == layerIndex) { inQueueOrLoading = true; break; }
            }
        }
        
        if (inQueueOrLoading) {
            LOGI("PrepareLayer: HIT prefetch/queue for layer %d. Waiting for I/O...", layerIndex);
            
            // CRITICAL FIX: Unlock 'mutex' before waiting, otherwise prefetch thread 
            // will deadlock when it tries to lock 'mutex' to finish loading.
            // But we need to keep 'prefetchMutex' locked for the wait.
            // We use a temporary unlocker for 'mutex' or just reduce its scope.
            
            // Since 'mutex' (the lock from line 30) is held, we MUST unlock it.
            lock.unlock(); 
            
            prefetchCv.wait(pfLock, [this, layerIndex] { 
                return loadedLayers.count(layerIndex) > 0 || !prefetchRunning; 
            });
            
            // Re-lock 'mutex' after waking up to maintain class invariants
            lock.lock();
            
            if (loadedLayers.count(layerIndex)) return true;
        }
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
            // CRITICAL: Never evict the current compute layer
            if (loaded == activeComputeLayer) continue;
            
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
                 // Force evict ANY layer that isn't current
                 for (int loaded : loadedLayers) {
                     if (loaded != activeComputeLayer) {
                         void* data = weightBuffers[loaded]->getData();
                         size_t size = weightBuffers[loaded]->getSize();
                         if (data) madvise(data, size, MADV_DONTNEED);
                         weightBuffers.erase(loaded);
                         loadedLayers.erase(loaded);
                         LOGI("Fallback Evicted layer %d (Hinted OS)", loaded);
                         break; 
                     }
                 }
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
    
    // HINT: We will read this layer sequentially
    madvise(lm.fullMapPtr, lm.fullMapSize, MADV_SEQUENTIAL);
    
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
    if (prefetchRunning) return;
    prefetchRunning = true;
    currentPrefetchingLayer = -1;
    prefetchQueue.clear();
    prefetchThread = std::thread(&LayerScheduler::prefetchThreadLoop, this);
    LOGI("LayerScheduler: Prefetcher thread started.");
}

void LayerScheduler::stopPrefetcher() {
    if (!prefetchRunning) return;
    prefetchRunning = false;
    {
        std::lock_guard<std::mutex> lock(prefetchMutex);
        prefetchQueue.clear();
        currentPrefetchingLayer = -2; // Exit signal
    }
    prefetchCv.notify_all();
    if (prefetchThread.joinable()) {
        prefetchThread.join();
    }
    LOGI("LayerScheduler: Prefetcher thread stopped.");
}

void LayerScheduler::queuePrefetch(int layerIndex) {
    if (!prefetchRunning) return;
    if (maxLayersInMemory < 2) return;
    
    {
        std::lock_guard<std::mutex> lock(prefetchMutex);
        // Avoid duplicates in queue or current loading
        if (loadedLayers.count(layerIndex) || currentPrefetchingLayer == layerIndex) return;
        for (int q : prefetchQueue) if (q == layerIndex) return;
        
        prefetchQueue.push_back(layerIndex);
    }
    prefetchCv.notify_one();
}

void LayerScheduler::prefetchThreadLoop() {
    while (prefetchRunning) {
        int target = -1;
        {
            std::unique_lock<std::mutex> lock(prefetchMutex);
            prefetchCv.wait(lock, [this] { return !prefetchQueue.empty() || !prefetchRunning; });
            
            if (!prefetchRunning || currentPrefetchingLayer == -2) break;
            
            target = prefetchQueue.front();
            prefetchQueue.pop_front();
            currentPrefetchingLayer = target;
        }

        if (target != -1) {
            LOGI("Prefetcher: Starting background load for layer %d", target);
            bool success = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!loadedLayers.count(target)) {
                    // EVICTION LOGIC with Multi-Layer Protection
                    if (loadedLayers.size() >= maxLayersInMemory) {
                        int evictCandidate = -1;
                        
                        // Set of protected layers: active + queue
                        std::set<int> protectedLayers;
                        protectedLayers.insert(activeComputeLayer);
                        {
                            std::lock_guard<std::mutex> qLock(prefetchMutex);
                            for (int q : prefetchQueue) protectedLayers.insert(q);
                        }

                        // Candidates: layers not in protected set
                        for (int loaded : loadedLayers) {
                            if (protectedLayers.count(loaded)) continue;
                            
                            // Prefer evicting layers behind the active one
                            if (loaded < activeComputeLayer) {
                                if (evictCandidate == -1 || loaded < evictCandidate) evictCandidate = loaded;
                            }
                        }
                        
                        // Fallback: evict any non-protected
                        if (evictCandidate == -1) {
                            for (int loaded : loadedLayers) {
                                if (!protectedLayers.count(loaded)) {
                                    evictCandidate = loaded;
                                    break;
                                }
                            }
                        }
                        
                        if (evictCandidate != -1) {
                            void* data = weightBuffers[evictCandidate]->getData();
                            size_t size = weightBuffers[evictCandidate]->getSize();
                            if (data) madvise(data, size, MADV_DONTNEED);
                            weightBuffers.erase(evictCandidate);
                            loadedLayers.erase(evictCandidate);
                            LOGI("Prefetcher: Evicted layer %d", evictCandidate);
                        }
                    }
                    success = loadLayerInternal(target);
                } else {
                    success = true;
                }
            }

            {
                std::lock_guard<std::mutex> lock(prefetchMutex);
                currentPrefetchingLayer = -1; 
            }
            prefetchCv.notify_all(); // Wake up any waiters in prepareLayer
            
            if (success) {
                LOGI("Prefetcher: Finished loading layer %d. Touching memory...", target);
                
                void* data = nullptr;
                size_t size = 0;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (weightBuffers.count(target)) {
                        data = weightBuffers[target]->getData();
                        size = weightBuffers[target]->getSize();
                    }
                }
                
                if (data && size > 0) {
                    volatile uint8_t* p = static_cast<volatile uint8_t*>(data);
                    size_t step = 4096;
                    for (size_t i = 0; i < size; i += step) {
                        (void)p[i];
                    }
                    LOGI("Prefetcher: Memory touch complete for layer %d.", target);
                }
            }
        }
    }
}
