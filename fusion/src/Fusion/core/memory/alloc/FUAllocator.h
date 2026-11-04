// -----------------------------------------------------------------------------
// FUAllocator invariants / current behavior
//
// Invariants:
// - chunks_ is indexed by ChunkID; chunk_id is expected to match the index into
//   chunks_ for all live and decpricated chunks.
// - A chunk with size == 0 is considered depricated and must have:
//     ptr == nullptr
//     requested_size == 0
//     in_use == false
//     prev == kInvalidChunkID
//     next == kInvalidChunkID
// - A non-deleted chunk must have ptr != nullptr.
// - Free chunks must have in_use == false and requested_size == 0.
// - Allocated chunks must have in_use == true.
// - merge_chunks(left, right) is directional:
//     left survives, right is depricated.
// - Coalescing is only valid for physically adjacent chunks; prev/next linkage
//   alone is not sufficient.
// - Buckets contain only free, non-deleted chunks and are keyed by
//   round_down_pow2(chunk.size).
// - RegionManager::ptr_chunk_map_ must map live chunk base pointers to the
//   current owning ChunkID.
// - On merge, the right chunk pointer mapping must be erased.
//
// Current design notes:
// - Region growth policy is conservative: repeated small allocations may create
//   separate root regions instead of sibling chunks from a shared larger slab.
// - Coalescing currently works within a grown region when adjacent split
//   siblings become free.
// - Double free detection is implemented via chunk.in_use checks in
// deallocate().
//
// TODOs:
// - Revisit growth policy so small allocations are more often sourced from
//   larger reusable slabs.
// - Separate "request rounding policy" from "region growth policy".
// - Add a helper for mergeability (e.g. adjacency + free-state) to avoid
//   duplicating merge precondition logic.
// - Consider adding a debug_validate() routine for allocator structural
//   invariants, bucket consistency, and non-overlap checks.
// - Revisit whether requested_size should store the original user request or
//   the rounded allocation size.
// - Review bucket cleanup / structural consistency under more aggressive fuzz
//   and fragmentation scenarios.
// -----------------------------------------------------------------------------

#ifndef FUSION_ALLOC_FU_ALLOCATOR_H
#define FUSION_ALLOC_FU_ALLOCATOR_H

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

class FUAllocator final : public IAllocator {
 public:
   FUAllocator();
   ~FUAllocator() override;

   FUAllocator(const FUAllocator &) = delete;
   FUAllocator &operator=(const FUAllocator &) = delete;
   FUAllocator(FUAllocator &&) noexcept = delete;
   FUAllocator &operator=(FUAllocator &&) noexcept = delete;

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
   ChunkID leftmost_mergeable_chunk(ChunkID chunk_id);

   std::unique_ptr<ISubAllocator> sub_allocator_;
   std::vector<Chunk> chunks_;
   RegionManager region_manager_;
   std::map<std::size_t, Bucket> buckets_by_size_;

   std::size_t current_allocation_size_ = kMinAllocationSize;
   std::size_t chunk_counter_ = 0;
};

#endif // FUSION_ALLOC_FU_ALLOCATOR_H