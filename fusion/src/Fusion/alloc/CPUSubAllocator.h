#ifndef CPU_SUB_INTERFACE_H
#define CPU_SUB_INTERFACE_H

#include <cstddef>
#include <cstdlib>
#include <stdexcept>

#include "AllocTypes.h"
#include "SubAllocatorInterface.h"

/* This file contains the CPUSubAllocator class/helpers, the purpose of this
 * class is to reserve a large chunk of aligned contigous memory - for example
 * 2mb with 64 bit alighment.*/

inline void *aligned_alloc_region(Alignment alignment, std::size_t size) {
   if (alignment < alignof(void *) || (alignment & (alignment - 1)) != 0) {
      throw std::invalid_argument(
          "alignment must be a power of two and >= alignof(void*)");
   }
   // TODO: add windows to if/def
   void *ptr = nullptr; // NOLINT
#if defined(__APPLE__) || defined(__linux__)
   const int rc = posix_memalign(&ptr, alignment, size); // returns 0 on success
   if (rc != 0 || ptr == nullptr) {
      throw std::bad_alloc();
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

   void *allocate_region(Alignment alignment, std::size_t size_bytes) override {
      void *ptr = aligned_alloc_region(alignment, size_bytes); // NOLINT
      return ptr;
   };
   void deallocate_region(void *ptr) override {
      std::free(ptr); // NOLINT
   };
};

#endif // CPU_SUB_INTERFACE_H
