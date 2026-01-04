// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef SUB_ALLOCATOR_INTERFACE_H
#define SUB_ALLOCATOR_INTERFACE_H

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

#endif // SUB_ALLOCATOR_INTERFACE_H
