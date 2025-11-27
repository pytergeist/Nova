#ifndef DEFAULT_ALLOCATOR_H
#define DEFAULT_ALLOCATOR_H
#pragma once

#include "AllocatorInterface.h"
#include "BFCPoolAllocator.h"

inline IAllocator &default_allocator() {
   static PoolAllocator pool;
   return pool;
};

#endif // DEFAULT_ALLOCATOR_H
