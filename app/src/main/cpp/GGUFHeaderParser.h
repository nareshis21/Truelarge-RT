#ifndef GGUF_HEADER_PARSER_H
#define GGUF_HEADER_PARSER_H

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// Structure to hold offset and size of a layer's tensor data
// Structure to hold metadata for a single tensor
struct TensorInfo {
    std::string name;
    size_t offset;
    size_t size;
    std::vector<int64_t> dims;
    uint32_t type; // ggml_type enum
};

// Structure to hold all tensors for a single layer
struct LayerSourceInfo {
    int index;
    // Map of tensor suffix (e.g. "attn_q.weight") to TensorInfo
    std::map<std::string, TensorInfo> tensors;
};

class GGUFHeaderParser {
public:
    GGUFHeaderParser(const std::string& modelPath);
    ~GGUFHeaderParser();

    // Parse the GGUF file header and populate layerMap
    bool parse();

    // Get layer source info by index
    const LayerSourceInfo* getLayerSourceInfo(int layerIndex) const;

    // Get total number of layers found
    int getLayerCount() const;
    
    // Debug print
    void printLayerMap() const;

private:
    std::string modelPath;
    int fd;
    void* mappedHeader;
    size_t fileSize;
    
    // Map of Layer Index -> LayerSourceInfo
    std::map<int, LayerSourceInfo> layerMap; 
    
    // Helper to read data from mapped memory
    template<typename T>
    T read(size_t& offset);
    
    std::string readString(size_t& offset);
    
    // Helper to identify layer index and extract suffix from tensor name
    // Returns index, populates suffix. Returns -1 if not a layer tensor.
    int parseTensorName(const std::string& name, std::string& suffix);

    // Helper to skip a GGUF value based on type
    void skipValue(uint32_t type, size_t& offset);
};

#endif // GGUF_HEADER_PARSER_H
