#include "WeightBuffer.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

WeightBuffer::WeightBuffer() : buffer(nullptr), size(0) {}

WeightBuffer::~WeightBuffer() {
    release();
}

bool WeightBuffer::allocate(size_t newSize) {
    if (buffer != nullptr) {
        if (size >= newSize) {
            // Re-use existing buffer if large enough? 
            // For safety and strict memory accounting, let's re-allocate or just return true if exactly same?
            // Better to release and re-allocate to ensure clean state, or just return false if already alloc.
            // Let's assume we reuse if size matches, else realloc.
            if (size == newSize) return true;
            release();
        } else {
            release();
        }
    }

    try {
        // Use aligned_alloc or just malloc? ggml often likes alignment.
        // ggml_tensor data usually 32-byte aligned.
        #ifdef __ANDROID__
            // Android malloc is usually 8 or 16 byte aligned.
            // verifying posix_memalign availability or use malloc.
            // Simple malloc for now.
             buffer = malloc(newSize);
        #else
             buffer = malloc(newSize);
        #endif

        if (!buffer) {
            std::cerr << "WeightBuffer: Failed to allocate " << newSize << " bytes" << std::endl;
            return false;
        }
        size = newSize;
        return true;
    } catch (const std::bad_alloc& e) {
        std::cerr << "WeightBuffer: Allocation exception: " << e.what() << std::endl;
        return false;
    }
}

void WeightBuffer::loadFrom(const void* src, size_t copySize) {
    if (!buffer) {
        std::cerr << "WeightBuffer: Attempt to load into null buffer" << std::endl;
        return;
    }
    if (copySize > size) {
        std::cerr << "WeightBuffer: Copy size " << copySize << " exceeds buffer size " << size << std::endl;
        return;
    }
    if (!src) {
         std::cerr << "WeightBuffer: Source is null" << std::endl;
         return;
    }

    // This memcpy captures the data into our explicit RAM buffer.
    // If src is mmapped, this triggers the page faults.
    std::memcpy(buffer, src, copySize);
}

void WeightBuffer::loadAt(size_t offset, const void* src, size_t copySize) {
    if (!buffer) return;
    if (offset + copySize > size) {
        std::cerr << "WeightBuffer: Overflow at offset " << offset << std::endl;
        return;
    }
    char* dest = static_cast<char*>(buffer) + offset;
    std::memcpy(dest, src, copySize);
}

void WeightBuffer::release() {
    if (buffer) {
        free(buffer);
        buffer = nullptr;
    }
    size = 0;
}
