
#include "CPUSubAllocator.h"

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

void *CPUSubAllocator::allocate_region(Alignment alignment,
                                       std::size_t size_bytes) {
   void *ptr = aligned_alloc_region(alignment, size_bytes); // NOLINT
   return ptr;
};

void CPUSubAllocator::deallocate_region(void *ptr) {
   std::free(ptr); // NOLINT
};
