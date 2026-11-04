#ifndef FUSION_ALLOC_SUB_ALLOCATOR_INTERFACE_H
#define FUSION_ALLOC_SUB_ALLOCATOR_INTERFACE_H

#include <cstddef>

#include "AllocTypes.h"

class ISubAllocator {
 public:
   virtual ~ISubAllocator() = default;

   ISubAllocator() = default;
   ISubAllocator(const ISubAllocator &) = delete;
   ISubAllocator &operator=(const ISubAllocator &) = delete;
   ISubAllocator(ISubAllocator &&) noexcept = delete;
   ISubAllocator &operator=(ISubAllocator &&) noexcept = delete;

   virtual void *allocate_region(Alignment alignment,
                                 std::size_t size_bytes) = 0;
   virtual void deallocate_region(void *ptr) = 0;
};

#endif // FUSION_ALLOC_SUB_ALLOCATOR_INTERFACE_H
