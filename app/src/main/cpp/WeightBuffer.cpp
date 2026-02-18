#include <sys/mman.h>
#include "WeightBuffer.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <new>

WeightBuffer::WeightBuffer() : buffer(nullptr), size(0), mapPtr(nullptr), mapSize(0), isMmap(false) {}

WeightBuffer::~WeightBuffer() {
    release();
}

bool WeightBuffer::allocate(size_t newSize) {
    if (buffer != nullptr) {
        if (size == newSize && !isMmap) return true;
        release();
    }

    buffer = malloc(newSize);
    if (!buffer) {
        return false;
    }
    size = newSize;
    isMmap = false;
    return true;
}

void WeightBuffer::adoptMmap(void* ptr, size_t newSize, void* rMapPtr, size_t rMapSize) {
    release();
    buffer = ptr;
    size = newSize;
    mapPtr = rMapPtr;
    mapSize = rMapSize;
    isMmap = true;
}

void WeightBuffer::loadFrom(const void* src, size_t copySize) {
    if (!buffer || isMmap) return;
    if (copySize > size || !src) return;
    std::memcpy(buffer, src, copySize);
}

void WeightBuffer::release() {
    if (buffer) {
        if (isMmap) {
            munmap(mapPtr, mapSize);
        } else {
            free(buffer);
        }
        buffer = nullptr;
    }
    size = 0;
    mapPtr = nullptr;
    mapSize = 0;
    isMmap = false;
}
