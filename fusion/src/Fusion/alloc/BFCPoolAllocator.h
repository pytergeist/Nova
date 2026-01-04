// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H

#include <array>
#include <bit>
#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "Fusion/common/Checks.hpp"
#include "Fusion/common/Log.hpp"

#include "AllocTypes.h"
#include "AllocatorInterface.h"
#include "CPUSubAllocator.h"
#include "Pool.h"

class CPUSubAllocator;

static constexpr std::size_t kMinAllocationSize = 64;

struct Region {
   void *ptr;
   std::size_t region_id;
   std::size_t size;
   Alignment alignment;
};

class RegionManager {
 public:
   RegionManager() = default;

   void add_allocated_region(void *ptr, std::size_t region_size,
                             Alignment alignment);

   Region &find_region_for_ptr(void *ptr);

   ChunkID get_chunkid_from_ptr(void *chunk_ptr);
   void set_chunkid(void *chunk_ptr, ChunkID chunk_id);
   bool erase_chunk(void *chunk_ptr);

   std::vector<Region> regions() const;
   std::vector<Region> regions();

 private:
   std::unordered_map<void *, ChunkID> ptr_chunk_map_;
   std::vector<Region> regions_;
   std::size_t counter_ = 0;
};

class PoolAllocator final : public IAllocator {
 public:
   PoolAllocator();
   ~PoolAllocator() override;

   PoolAllocator(const PoolAllocator &) = delete;
   PoolAllocator &operator=(const PoolAllocator &) = delete;
   PoolAllocator(PoolAllocator &&) noexcept = delete;
   PoolAllocator &operator=(PoolAllocator &&) noexcept = delete;

   void *allocate(std::size_t size, Alignment alignment) override;
   void deallocate(void *ptr) override;

   std::vector<Chunk> chunks() const;
   std::vector<ChunkID> get_free_chunks(std::size_t bucket_size) const;

 private:
   static std::size_t round_up_pow2(std::size_t n);
   static std::size_t round_down_pow2(std::size_t n);

   Chunk &get_chunk_from_id(ChunkID chunk_id);
   Bucket &get_or_create_bucket(std::size_t bucket_size);

   ChunkID find_free_chunk_id_for_size(std::size_t size);
   void grow_pool_for_size(std::size_t size, Alignment alignment);

   void *allocate_bucket_region(std::size_t region_size, Alignment alignment);

   ChunkID split_chunk_for_allocation(ChunkID chunk_id, std::size_t size);

   static void delete_chunk(Chunk &chunk);
   void erase_chunk_from_bucket(Chunk &chunk);

   ChunkID merge_chunks(Chunk &left, Chunk &right);
   ChunkID free_and_maybe_coalesce(ChunkID chunk_id);

 private:
   std::unique_ptr<ISubAllocator> sub_allocator_;
   std::vector<Chunk> chunks_;
   RegionManager region_manager_;
   std::map<std::size_t, Bucket> buckets_by_size_;

   std::size_t current_allocation_size_ = kMinAllocationSize;
   std::size_t chunk_counter_ = 0;
};

#endif // POOL_ALLOCATOR_H
