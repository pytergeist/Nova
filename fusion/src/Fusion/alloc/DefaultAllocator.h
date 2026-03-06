#ifndef DEFAULT_ALLOCATOR_H
#define DEFAULT_ALLOCATOR_H

#include "AllocatorInterface.h"
#include "FUAllocator.h"

// TODO: this will be refactored to be AllocContext on addition of scope
// specific allocators such as Arena (for autodiff cst) & Slab (for physics
// sims)

inline IAllocator &default_allocator() {
   static FUAllocator pool;
   return pool;
};

#endif // DEFAULT_ALLOCATOR_H
