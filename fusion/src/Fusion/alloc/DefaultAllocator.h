#ifndef DEFAULT_ALLOCATOR_H
#define DEFAULT_ALLOCATOR_H
#pragma once

#include "AllocatorInterface.h"
#include "BFCPoolAllocator.h"

// TODO: this will be refactored to be AllocContext on addition of scope specific allocators
// such as Arena (for autodiff cst) & Slab (for physics sims)

inline IAllocator &default_allocator() {
   static PoolAllocator pool;
   return pool;
};

#endif // DEFAULT_ALLOCATOR_H
