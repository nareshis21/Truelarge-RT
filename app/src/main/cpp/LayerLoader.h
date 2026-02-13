#ifndef LAYER_LOADER_H
#define LAYER_LOADER_H

#include <string>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

class LayerLoader {
public:
    LayerLoader(const std::string& path);
    ~LayerLoader();

    // Initialize file descriptor
    bool init();

    // Map a specific layer into memory (Virtual Sharding)
    // Returns pointer to the mapped memory
    void* loadLayer(size_t offset, size_t size);

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
