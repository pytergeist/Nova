#ifndef FUSION_ALLOC_ALLOCATOR_INTERFACE_H
#define FUSION_ALLOC_ALLOCATOR_INTERFACE_H

#include <memory>

#include "AllocTypes.h"

class IAllocator {
 public:
   virtual ~IAllocator() = default;

   IAllocator() = default;
   IAllocator(const IAllocator &) = delete;
   IAllocator &operator=(const IAllocator &) = delete;
   IAllocator(IAllocator &&) noexcept = delete;
   IAllocator &operator=(IAllocator &&) noexcept = delete;

   virtual void *allocate(std::size_t size, Alignment alignment) = 0;
   virtual void deallocate(void *p) = 0;
};

#endif // FUSION_ALLOC_ALLOCATOR_INTERFACE_H
