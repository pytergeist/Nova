#ifndef CPU_SUB_INTERFACE_H
#define CPU_SUB_INTERFACE_H
#pragma once

#include <cstddef>

#include "SubAllocatorInterface.h"

/* This file contains the CPUSubAllocator class/helpers, the purpose of this class
 * is to reserve a large chunk of aligned contigous memory - for example 2mb with
 * 64 bit alighment.*/

inline void *aligned_alloc_region(std::size_t alignment, std::size_t size) {
    if (alignment < alignof(void *) || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(
            "alignment must be a power of two and >= alignof(void*)");
    }
    // TODO: add windows to if/def
    void *ptr = nullptr;
    #if defined(__APPLE__) || defined(__linux__)
    int rc = posix_memalign(&ptr, alignment, size); // returns 0 on success
    if (rc != 0 || !ptr) {
        throw std::bad_alloc(); // TODO: maybe return nullptr here and handle with retry
    }
    return ptr;
    #else
      ptr = std::aligned_alloc(alignment, size);
      return ptr;
    #endif
};

class CPUSubAllocator : public ISubAllocator {
public:
    CPUSubAllocator() = default;
    ~CPUSubAllocator() = default;

    void* allocate_region(std::size_t alignment, std::size_t size_bytes) override {
        // returns ptr to beginning of allocated memory region
        void* ptr = aligned_alloc_region(alignment, size_bytes);
        return ptr;
    };
    void deallocate_region(void* ptr) override {
        // TODO: does this ptr need to be the ptr to the beginning of mem region
        std::free(ptr);
    };

};

#endif // CPU_SUB_INTERFACE_H
