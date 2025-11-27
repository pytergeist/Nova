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

#include "../common/Checks.h"
#include "../common/Log.h"
#include "AllocatorInterface.h"
#include "CPUSubAllocator.h"
#include "Pool.h"

static constexpr std::size_t kMinAllocationSize = 64;

struct Region {
   void *ptr;
   std::size_t region_id;
   std::size_t size;
   std::size_t alignment;
};

class RegionManager {
 public:
   RegionManager() = default;
   ~RegionManager() = default;

   void add_allocated_region(void *ptr, std::size_t region_size,
                             std::size_t alignment) {
      regions_.emplace_back(Region{ptr, counter_, region_size, alignment});
      counter_++;
   }

   Region &find_region_for_ptr(void *ptr) {
      auto addr = reinterpret_cast<std::byte *>(ptr);
      for (auto &region : regions_) {
         auto base = reinterpret_cast<std::byte *>(region.ptr);
         auto end = base + region.size;
         if (addr >= base && addr < end) {
            return region;
         }
      }
      throw std::runtime_error("RegionManager: failed to find region for ptr");
   }

   ChunkId get_chunkid_from_ptr(void *chunk_ptr) {
      auto it = ptr_chunk_map_.find(chunk_ptr);
      if (it == ptr_chunk_map_.end()) {
         FUSION_LOGI(false,
                     "PoolAllocator: deallocate called with unknown pointer: ",
                     chunk_ptr, " (double free or foreign pointer)");
         throw std::runtime_error(
             "PoolAllocator: unknown pointer in get_chunkid_from_ptr");
      }
      return it->second;
   }

   void set_chunkid(void *chunk_ptr, std::size_t chunk_id) {
      // Reuses same ptr for different chunkIds after coalesce / split
      ptr_chunk_map_.insert_or_assign(chunk_ptr, chunk_id);
   }

   bool erase_chunk(void *chunk_ptr) {
      // returns 1 if existed, 0 if not
      return static_cast<bool>(ptr_chunk_map_.erase(chunk_ptr));
   }

   std::vector<Region> regions() { return regions_; }
   std::vector<Region> regions() const { return regions_; }

 private:
   std::unordered_map<void *, ChunkId> ptr_chunk_map_;
   std::vector<Region> regions_;
   std::size_t counter_ = 0;
};

class PoolAllocator : public IAllocator {
 public:
   PoolAllocator() : sub_allocator_(std::make_unique<CPUSubAllocator>()) {}

   ~PoolAllocator() override = default;

   void *allocate(std::size_t size, std::size_t alignment) override {
      if (size == 0) {
         size = 1;
      }

      size = round_up_pow2(size);

      ChunkId free_id = find_free_chunk_id_for_size(size);

      if (free_id == kInvalidBucketId) {
         grow_pool_for_size(size, alignment);
         free_id = find_free_chunk_id_for_size(size);
         if (free_id == kInvalidBucketId) {
            // TODO: change back to bad_alloc
            throw std::runtime_error("Allocate: free_id is invalid bucket");
         }
      }

      Chunk &chunk = get_chunk_from_id(free_id);

      erase_chunk_from_bucket(chunk);

      ChunkId allocated_id = split_chunk_for_allocation(chunk.chunk_id, size);
      Chunk &allocated = get_chunk_from_id(allocated_id);

      allocated.in_use = true;
      allocated.requested_size = size;

      void *chunk_ptr = allocated.ptr;
      if (chunk_ptr != nullptr) {
         return chunk_ptr;
      }
      // TODO: change back to bad_alloc
      throw std::runtime_error("PoolAllocator: Chunk Ptr is null");
   }

   void deallocate(void *ptr) override {
      if (!ptr) {
         return;
      }

      ChunkId chunk_id = region_manager_.get_chunkid_from_ptr(ptr);
      Chunk &chunk = get_chunk_from_id(chunk_id);

      chunk.in_use = false;
      chunk.requested_size = 0;

      chunk_id = free_and_maybe_coalesce(chunk_id);
      Chunk &merged = get_chunk_from_id(chunk_id);

      std::size_t bucket_size = round_down_pow2(merged.size);
      Bucket &bucket = get_or_create_bucket(bucket_size);
      bucket.free_chunks.insert(chunk_id);
   }

   std::vector<Chunk> chunks() { return chunks_; }

   std::set<ChunkId> get_free_chunks(std::size_t bucket_size) {
      auto it = buckets_by_size_.find(bucket_size);
      if (it == buckets_by_size_.end()) {
         return {};
      }
      return it->second.free_chunks;
   }

 private:
   std::unique_ptr<ISubAllocator> sub_allocator_;
   std::vector<Chunk> chunks_;
   RegionManager region_manager_;

   std::map<std::size_t, Bucket> buckets_by_size_;

   std::size_t current_allocation_size_ = kMinAllocationSize;
   std::size_t chunk_counter_ = 0;

   static std::size_t round_up_pow2(std::size_t n) {
      if (n <= 1)
         return 1;
      int bw = std::bit_width(n - 1);
      if (bw >= int(sizeof(std::size_t) * 8)) {
         return std::numeric_limits<std::size_t>::max();
      }
      return std::size_t{1} << bw;
   }

   static std::size_t round_down_pow2(std::size_t n) {
      if (n <= 1)
         return 1;
      int bw = std::bit_width(n);
      return std::size_t{1} << (bw - 1);
   }

   Chunk &get_chunk_from_id(ChunkId chunk_id) {
      FUSION_BOUNDS_CHECK(chunk_id, chunks_.size());
      return chunks_[chunk_id];
   }

   Bucket &get_or_create_bucket(std::size_t bucket_size) {
      auto it = buckets_by_size_.find(bucket_size);
      if (it != buckets_by_size_.end()) {
         return it->second;
      }

      Bucket bucket{};
      bucket.bucket_size = bucket_size;
      bucket.bucket_id = static_cast<BucketId>(buckets_by_size_.size());

      auto [inserted_it, _] =
          buckets_by_size_.emplace(bucket_size, std::move(bucket));
      return inserted_it->second;
   }

   ChunkId find_free_chunk_id_for_size(std::size_t size) {
      std::size_t size_class = round_up_pow2(size);

      for (auto it = buckets_by_size_.lower_bound(size_class);
           it != buckets_by_size_.end(); ++it) {
         Bucket &bucket = it->second;
         if (bucket.free_chunks.empty()) {
            continue;
         }
         for (ChunkId id : bucket.free_chunks) {
            Chunk &c = get_chunk_from_id(id);
            if (!c.in_use && c.size >= size) {
               return id;
            }
         }
      }
      return kInvalidBucketId;
   }

   void grow_pool_for_size(std::size_t size, std::size_t alignment) {
      while (current_allocation_size_ < size) {
         current_allocation_size_ <<= 1;
      }

      void *ptr = allocate_bucket_region(current_allocation_size_, alignment);

      Chunk chunk;
      chunk.ptr = ptr;
      chunk.chunk_id = static_cast<ChunkId>(chunk_counter_++);
      chunk.prev = kInvalidChunkId;
      chunk.next = kInvalidChunkId;
      chunk.size = current_allocation_size_;
      chunk.in_use = false;
      chunk.requested_size = 0;
      chunk.set_end_ptr();

      region_manager_.set_chunkid(ptr, chunk.chunk_id);
      chunks_.push_back(chunk);

      std::size_t bucket_size = round_down_pow2(chunk.size);
      Bucket &bucket = get_or_create_bucket(bucket_size);
      bucket.free_chunks.insert(chunk.chunk_id);
   }

   void *allocate_bucket_region(std::size_t region, std::size_t alignment) {
      void *ptr = sub_allocator_->allocate_region(alignment, region);
      region_manager_.add_allocated_region(ptr, region, alignment);
      return ptr;
   }

   ChunkId split_chunk_for_allocation(ChunkId chunk_id, std::size_t size) {

      Chunk &chunk = get_chunk_from_id(chunk_id);

      if (chunk.size < size) {
         throw std::runtime_error(
             "split_chunk_for_allocation: chunk too small");
      }

      std::size_t remainder_size = chunk.size - size;

      if (remainder_size < kMinAllocationSize) {
         return chunk_id;
      }

      std::byte *base = static_cast<std::byte *>(chunk.ptr);
      void *rem_ptr = static_cast<void *>(base + size);

      Chunk remainder;
      remainder.ptr = rem_ptr;
      remainder.chunk_id = static_cast<ChunkId>(chunk_counter_++);
      remainder.prev = chunk.chunk_id;
      remainder.next = chunk.next;
      remainder.size = remainder_size;
      remainder.in_use = false;
      remainder.requested_size = 0;
      remainder.set_end_ptr();

      if (chunk.next != kInvalidChunkId) {
         Chunk &next_chunk = get_chunk_from_id(chunk.next);
         next_chunk.prev = remainder.chunk_id;
      }
      chunk.next = remainder.chunk_id;

      chunk.size = size;
      chunk.set_end_ptr();
      region_manager_.set_chunkid(remainder.ptr, remainder.chunk_id);
      chunks_.push_back(remainder);

      std::size_t rem_bucket_size = round_down_pow2(remainder.size);
      Bucket &rem_bucket = get_or_create_bucket(rem_bucket_size);
      rem_bucket.free_chunks.insert(remainder.chunk_id);

      return chunk_id;
   }

   void delete_chunk(Chunk &chunk) {
      chunk.size = 0;
      chunk.requested_size = 0;
      chunk.ptr = nullptr;
      chunk.end_ptr_ = nullptr;
      chunk.prev = kInvalidChunkId;
      chunk.next = kInvalidChunkId;
      chunk.in_use = false;
   }

   void erase_chunk_from_bucket(Chunk &chunk) {
      if (chunk.size == 0) {
         return;
      }
      std::size_t bucket_size = round_down_pow2(chunk.size);
      auto it = buckets_by_size_.find(bucket_size);
      if (it == buckets_by_size_.end()) {
         return;
      }
      Bucket &bucket = it->second;
      bucket.free_chunks.erase(chunk.chunk_id);
   }

   ChunkId merge_chunks(Chunk &lchunk, Chunk &rchunk) {
      std::byte *lbase = static_cast<std::byte *>(lchunk.ptr);
      std::byte *rbase = static_cast<std::byte *>(rchunk.ptr);
      if (lbase + lchunk.size == rbase) {
         region_manager_.erase_chunk(rchunk.ptr);
         erase_chunk_from_bucket(rchunk);
         erase_chunk_from_bucket(lchunk);

         lchunk.size += rchunk.size;
         lchunk.set_end_ptr();

         ChunkId rnext_id = rchunk.next;
         lchunk.next = rnext_id;
         if (rnext_id != kInvalidChunkId) {
            Chunk &rnext = get_chunk_from_id(rnext_id);
            rnext.prev = lchunk.chunk_id;
         }

         delete_chunk(rchunk);
         return lchunk.chunk_id;
      }
      return rchunk.chunk_id;
   }

   ChunkId free_and_maybe_coalesce(ChunkId chunk_id) {
      ChunkId current_id = chunk_id;

      while (true) {
         Chunk &chunk = get_chunk_from_id(current_id);

         if (chunk.prev == kInvalidChunkId) {
            break;
         }

         Chunk &prev_chunk = get_chunk_from_id(chunk.prev);
         if (prev_chunk.in_use) {
            break;
         }

         ChunkId new_id = merge_chunks(prev_chunk, chunk);

         if (new_id == current_id) {
            break;
         }

         current_id = new_id;
      }

      return current_id;
   }
};

#endif // POOL_ALLOCATOR_H
