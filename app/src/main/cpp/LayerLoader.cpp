#include "LayerLoader.h"
#include <sys/stat.h>
#include <stdexcept>
#include <cerrno>
#include <cstring>

LayerLoader::LayerLoader(const std::string& path) : filePath(path), fd(-1), fileSize(0) {
    pageSize = sysconf(_SC_PAGESIZE);
}

LayerLoader::~LayerLoader() {
    if (fd != -1) {
        close(fd);
    }
}

bool LayerLoader::init() {
    fd = open(filePath.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "LayerLoader: Failed to open file " << filePath << ": " << strerror(errno) << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "LayerLoader: Failed to stat file: " << strerror(errno) << std::endl;
        close(fd);
        fd = -1;
        return false;
    }
    fileSize = sb.st_size;
    return true;
}

void* LayerLoader::loadLayer(size_t offset, size_t size) {
    LayerMap lm = loadLayerMap(offset, size);
    return lm.data;
}

LayerMap LayerLoader::loadLayerMap(size_t offset, size_t size) {
    LayerMap lm;
    if (fd == -1) {
        return lm;
    }

    size_t alignedOffset = (offset / pageSize) * pageSize;
    size_t diff = offset - alignedOffset;
    size_t mapSize = size + diff;

    void* mapPtr = mmap(NULL, mapSize, PROT_READ, MAP_PRIVATE, fd, alignedOffset);

    if (mapPtr == MAP_FAILED) {
        return lm;
    }

    // Hint to kernel: read binary data into page cache ASAP
    madvise(mapPtr, mapSize, MADV_WILLNEED);

    lm.data = static_cast<char*>(mapPtr) + diff;
    lm.size = size;
    lm.fullMapPtr = mapPtr;
    lm.fullMapSize = mapSize;
    return lm;
}

void LayerLoader::unloadLayer(void* ptr, size_t size) {
    if (!ptr) return;

    // We need to calculate the original map pointer and size to munmap correctly
    // Since we returned (mapPtr + diff), we need to reverse it.
    // However, munmap usually handles unaligned pointers by rounding down, 
    // but strictly we should pass the pointer returned by mmap.
    
    // To do this cleanly, the caller or this class needs to track the 'diff'.
    // For simplicity here, we assume the caller passes back exactly what they got,
    // and we re-calculate alignment based on the pointer address itself which is tricky.
    
    // BETTER APPROACH: Return a struct or handle that contains the real map_ptr.
    // But for this implementation, we will assume standard behavior:
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t alignedAddr = (addr / pageSize) * pageSize;
    size_t diff = addr - alignedAddr;
    size_t mapSize = size + diff;

    if (munmap(reinterpret_cast<void*>(alignedAddr), mapSize) == -1) {
        std::cerr << "LayerLoader: munmap failed: " << strerror(errno) << std::endl;
    }
}
