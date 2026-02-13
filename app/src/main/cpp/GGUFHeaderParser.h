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
struct LayerInfo {
    size_t offset;
    size_t size;
    std::string name;
};

class GGUFHeaderParser {
public:
    GGUFHeaderParser(const std::string& modelPath);
    ~GGUFHeaderParser();

    // Parse the GGUF file header and populate layerMap
    bool parse();

    // Get layer info by index
    const LayerInfo* getLayerInfo(int layerIndex) const;

    // Get total number of layers found
    int getLayerCount() const;
    
    // Debug print
    void printLayerMap() const;

private:
    std::string modelPath;
    int fd;
    void* mappedHeader;
    size_t fileSize;
    
    // Map of Layer Index -> LayerInfo
    std::map<int, LayerInfo> layerMap; 
    
    // Helper to read data from mapped memory
    template<typename T>
    T read(size_t& offset);
    
    std::string readString(size_t& offset);
    
    // Helper to identify layer index from tensor name
    // Helper to identify layer index from tensor name
    int extractLayerIndex(const std::string& name);

    // Helper to skip a GGUF value based on type
    void skipValue(uint32_t type, size_t& offset);
};

#endif // GGUF_HEADER_PARSER_H
