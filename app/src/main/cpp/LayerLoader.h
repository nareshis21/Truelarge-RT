#ifndef LAYER_LOADER_H
#define LAYER_LOADER_H

#include <string>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

struct LayerMap {
    void* data = nullptr;     // Aligned pointer for usage
    size_t size = 0;          // Actual data size
    void* fullMapPtr = nullptr; // Original pointer from mmap
    size_t fullMapSize = 0;     // Original size used for mmap
};

class LayerLoader {
public:
    LayerLoader(const std::string& path);
    ~LayerLoader();

    // Initialize file descriptor
    bool init();

    // Map a specific layer into memory (Virtual Sharding)
    void* loadLayer(size_t offset, size_t size);

    // Get detailed mapping info (for adoption)
    LayerMap loadLayerMap(size_t offset, size_t size);

    // Unmap the layer to free memory immediately
    void unloadLayer(void* ptr, size_t size);

private:
    std::string filePath;
    int fd;
    size_t fileSize;
    
    // Page size for alignment
    long pageSize;
};

#endif // LAYER_LOADER_H
