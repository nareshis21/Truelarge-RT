#include "GGUFHeaderParser.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include "ggml.h"

// Define GGUF Magic "GGUF"
constexpr uint32_t GGUF_MAGIC = 0x46554747; 

enum GGUFType {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

GGUFHeaderParser::GGUFHeaderParser(const std::string& path) : modelPath(path), fd(-1), mappedHeader(nullptr), fileSize(0) {}

GGUFHeaderParser::~GGUFHeaderParser() {
    if (mappedHeader != nullptr && mappedHeader != MAP_FAILED) {
        munmap(mappedHeader, fileSize); 
    }
    if (fd != -1) {
        close(fd);
    }
}

template<typename T>
T GGUFHeaderParser::read(size_t& offset) {
    if (offset + sizeof(T) > fileSize) {
        throw std::runtime_error("Read out of bounds");
    }
    T val;
    std::memcpy(&val, static_cast<char*>(mappedHeader) + offset, sizeof(T));
    offset += sizeof(T);
    return val;
}

std::string GGUFHeaderParser::readString(size_t& offset) {
    uint64_t len = read<uint64_t>(offset);
    if (offset + len > fileSize) {
        throw std::runtime_error("String read out of bounds");
    }
    std::string str(static_cast<char*>(mappedHeader) + offset, len);
    offset += len;
    return str;
}

void GGUFHeaderParser::skipValue(uint32_t type, size_t& offset) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            offset += 1;
            break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            offset += 2;
            break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            offset += 4;
            break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            offset += 8;
            break;
        case GGUF_TYPE_STRING:
            readString(offset); // validation inside
            break;
        case GGUF_TYPE_ARRAY: {
            uint32_t itemType = read<uint32_t>(offset);
            uint64_t count = read<uint64_t>(offset);
            for (uint64_t i = 0; i < count; ++i) {
                skipValue(itemType, offset);
            }
            break;
        }
        default:
            throw std::runtime_error("Unknown GGUF type");
    }
}

// ... (imports remain)

    // ... (rest of methods)

bool GGUFHeaderParser::parse() {
    try {
        fd = open(modelPath.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "Failed to open file: " << modelPath << std::endl;
            return false;
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            close(fd);
            return false;
        }
        fileSize = sb.st_size;

        mappedHeader = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mappedHeader == MAP_FAILED) {
            close(fd);
            return false;
        }

        size_t offset = 0;

        // 1. Magic
        uint32_t magic = read<uint32_t>(offset);
        if (magic != GGUF_MAGIC) {
            std::cerr << "Invalid GGUF magic" << std::endl;
            return false;
        }

        // 2. Version
        uint32_t version = read<uint32_t>(offset);
        
        // 3. Tensor Count / KV Count
        uint64_t tensorCount = read<uint64_t>(offset);
        uint64_t metadataKVCount = read<uint64_t>(offset);
        
        std::cout << "GGUF Version: " << version << ", Tensors: " << tensorCount << ", KV: " << metadataKVCount << std::endl;

        // 4. Skip KV Pairs (Metadata)
        for (uint64_t i = 0; i < metadataKVCount; ++i) {
            readString(offset); // Key
            uint32_t valueType = read<uint32_t>(offset); // Type
            skipValue(valueType, offset); // Value
        }
        
        // 5. Tensor Info
        for (uint64_t i = 0; i < tensorCount; ++i) {
            std::string name = readString(offset);
            uint32_t n_dims = read<uint32_t>(offset);
            
            std::vector<int64_t> dims;
            uint64_t n_elements = 1;
            for (uint32_t j = 0; j < n_dims; ++j) {
                uint64_t dim = read<uint64_t>(offset);
                dims.push_back(dim);
                n_elements *= dim;
            }
            
            uint32_t type = read<uint32_t>(offset); // ggml_type
            uint64_t tensorOffset = read<uint64_t>(offset);
            
            std::string suffix;
            int layerIdx = parseTensorName(name, suffix);
            
            // Allow globals (index -1)
            if (layerIdx >= -1) {
                TensorInfo info;
                info.name = name;
                info.offset = tensorOffset; // Relative to data start
                info.dims = dims;
                info.type = type;
                
                // Calculate size using ggml logic
                size_t type_size = ggml_type_size((ggml_type)type);
                int64_t blck_size = ggml_blck_size((ggml_type)type);
                
                if (blck_size > 0) {
                     // Using integer arithmetic closely matches implementation
                     // Usually: elements * type_size / block_size
                     // NOTE: n_elements for quantized types (like Q4_K) is the number of "superblocks"? 
                     // No, n_elements is logical elements (parameters).
                     // ggml_type_size is bytes per block.
                     // ggml_blck_size is elements per block.
                     // size = (n_elements / blck_size) * type_size
                     info.size = (n_elements * type_size) / blck_size;
                } else {
                     info.size = 0; 
                }

                // Append to layer info
                layerMap[layerIdx].index = layerIdx;
                layerMap[layerIdx].tensors[suffix] = info;
            }
        }
        
        // Base offset for tensor data needs to be aligned
        size_t alignment = 32; 
        
        // Pad offset to alignment
        size_t padding = alignment - (offset % alignment);
        if (padding != alignment) {
            offset += padding;
        }
        
        // Now update all absolute offsets
        size_t dataStart = offset;
        for (auto& layerPair : layerMap) {
            for (auto& tensorPair : layerPair.second.tensors) {
                tensorPair.second.offset += dataStart;
            }
        }

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error parsing GGUF: " << e.what() << std::endl;
        return false;
    }
}

int GGUFHeaderParser::parseTensorName(const std::string& name, std::string& suffix) {
    // Expected format: blk.N.suffix
    // If not, returns -1 and suffix = name (Global)
    size_t run = name.find("blk.");
    if (run != std::string::npos) {
        size_t startNum = run + 4;
        size_t endNum = name.find('.', startNum);
        if (endNum != std::string::npos) {
            std::string num = name.substr(startNum, endNum - startNum);
            try {
                int index = std::stoi(num);
                suffix = name.substr(endNum + 1); // everything after "blk.N."
                return index;
            } catch (...) {
                // Fallthrough
            }
        }
    }
    
    // Global tensor
    suffix = name;
    return -1;
}

const LayerSourceInfo* GGUFHeaderParser::getLayerSourceInfo(int layerIndex) const {
    auto it = layerMap.find(layerIndex);
    if (it != layerMap.end()) {
        return &it->second;
    }
    return nullptr;
}

int GGUFHeaderParser::getLayerCount() const {
    return layerMap.size();
}

void GGUFHeaderParser::printLayerMap() const {
    for (const auto& pair : layerMap) {
        std::cout << "Layer " << pair.first << " has " << pair.second.tensors.size() << " tensors." << std::endl;
        for (const auto& tp : pair.second.tensors) {
             std::cout << "  - " << tp.first << " (" << tp.second.size << " bytes)" << std::endl;
        }
    }
}
