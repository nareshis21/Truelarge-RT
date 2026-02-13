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
    if (fd == -1) {
        std::cerr << "LayerLoader: File not initialized!" << std::endl;
        return nullptr;
    }

    // mmap requires offset to be aligned to page size
    size_t alignedOffset = (offset / pageSize) * pageSize;
    size_t diff = offset - alignedOffset;
    size_t mapSize = size + diff;

    // Use MAP_PRIVATE | MAP_POPULATE if available to pre-fault pages (optimization)
    // On Android MAP_POPULATE might not be available or needed effectively for separate layers
    // PROT_READ is sufficient.
    void* mapPtr = mmap(NULL, mapSize, PROT_READ, MAP_PRIVATE, fd, alignedOffset);

    if (mapPtr == MAP_FAILED) {
        std::cerr << "LayerLoader: mmap failed: " << strerror(errno) 
                  << " (Offset: " << offset << ", Size: " << size << ")" << std::endl;
        return nullptr;
    }

    // Advise OS that we will need this data (prefetch)
    // madvise(mapPtr, mapSize, MADV_WILLNEED);
    // madvise(mapPtr, mapSize, MADV_SEQUENTIAL);

    // Return pointer adjusted for alignment
    return static_cast<char*>(mapPtr) + diff;
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
