#ifndef WEIGHT_BUFFER_H
#define WEIGHT_BUFFER_H

#include <cstddef>
#include <cstdint>

/**
 * Manages a RAM buffer for a layer's weights.
 * Provides explicit control over allocation and deallocation to manage memory budget.
 */
class WeightBuffer {
public:
    WeightBuffer();
    ~WeightBuffer();

    // Disable copy
    WeightBuffer(const WeightBuffer&) = delete;
    WeightBuffer& operator=(const WeightBuffer&) = delete;

    // Allocate memory for the buffer
    bool allocate(size_t size);

    // Load data into the buffer from a source pointer
    // This typically triggers the page faults if src is mmapped
    void loadFrom(const void* src, size_t size);

    // Load data at specific offset (for packing tensors)
    void loadAt(size_t offset, const void* src, size_t size);

    // Release the memory
    void release();

    // Get pointer to data
    void* getData() const { return buffer; }
    
    // Get size of buffer
    size_t getSize() const { return size; }

    // Check if valid/allocated
    bool isValid() const { return buffer != nullptr; }

private:
    void* buffer = nullptr;
    size_t size = 0;
};

#endif // WEIGHT_BUFFER_H
