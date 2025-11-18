#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H
#include <cstddef>
#include <iostream>

#include "../common/Log.h"
#include "AllocatorInterface.h"
#include "BFCPool.h"
#include "CPUSubAllocator.h"

/* TODO: the below static array of bucket sizes is *fine* for an initialiser,
 * but we want to add new regions on request if necessary. Also need to figure
 * out if we should use dynamic bucket sizes and do coalescing. */

static constexpr std::size_t kNumBuckets = 30;

constexpr std::array<std::size_t, kNumBuckets> make_bucket_sizes() {
   std::array<std::size_t, kNumBuckets> sizes{};
   std::size_t value = 1;
   for (std::size_t i = 0; i < kNumBuckets; ++i) {
      sizes[i] = value;
      value <<= 1;
   }
   return sizes;
}

static constexpr std::array<std::size_t, kNumBuckets> kBucketSizes =
    make_bucket_sizes();

static constexpr std::size_t kChunkCount = 64;

class PoolAllocator : public IAllocator {
 public:
   PoolAllocator() : sub_allocator_(std::make_unique<CPUSubAllocator>()) {
      init_buckets();
   };
   ~PoolAllocator() = default;
   std::size_t chunk_counter = 0;

   void *allocate(std::size_t size, std::size_t alignment) override {
      if (size > kBucketSizes.back()) {
         throw std::bad_alloc();
      };
      std::size_t bucket_index = find_bucket_idx(size);
      Bucket &bucket = buckets_[bucket_index];
      if (!bucket.has_mem_attatched) {
         std::size_t region =
             calc_region_size(kBucketSizes[bucket_index], kChunkCount);
         void *ptr = allocate_bucket_region(region, alignment);
         bucket.ptr = ptr;
         bucket.region_size = region;
         bucket.has_mem_attatched = true;
         split_chunks(bucket_index, region / kChunkCount);
      }
      Chunk &chunk = find_free_chunk(bucket_index);
      void *chunk_ptr = chunk.ptr;
      chunk.in_use = true;
      chunk.requested_size = size;
      if (size > chunk.size) {
         throw std::bad_alloc();
      }
      if (chunk_ptr != nullptr) {
         return chunk_ptr;
      }
      throw std::bad_alloc();
   };

   void reset_chunk_metadata(Chunk &chunk) {
      chunk.in_use = false;
      chunk.requested_size = 0;
   }

   void deallocate(void *ptr) override {
      if (!ptr)
         return;
      Bucket &bucket = find_bucket_for_ptr(ptr);
      //        if (!bucket) {
      //           // TODO: err handle?
      //           return;
      //        }
      ChunkId id = chunk_idx_for_ptr(bucket, ptr);
      bucket.free_chunks.insert(id);
      reset_chunk_metadata(chunks_[id]);
   };

   std::vector<Chunk> get_chunks_list(std::size_t bucket_size) {
      std::size_t bucket_index = find_bucket_idx(bucket_size);
      std::vector<Chunk> chunk_list = chunks_;
      return chunk_list;
   }

   std::set<ChunkId> get_free_chunks(std::size_t bucket_size) {
      std::size_t bucket_index = find_bucket_idx(bucket_size);
      std::set<ChunkId> free_chunks_set = buckets_[bucket_index].free_chunks;
      return free_chunks_set;
   }

 private:
   std::unique_ptr<ISubAllocator> sub_allocator_;
   std::array<Bucket, kBucketSizes.size()> buckets_;
   std::vector<Chunk> chunks_;

   std::size_t calc_region_size(std::size_t bucket_size,
                                std::size_t chunk_count) {
      return chunk_count * bucket_size;
   }

   Chunk &find_free_chunk(std::size_t bucket_index) {
      Bucket &bucket =
          buckets_[bucket_index]; // TODO: mechanism if nothing in free list?
      ChunkId chunk_id = *bucket.free_chunks.begin();
      bucket.free_chunks.erase(bucket.free_chunks.begin());
      return chunks_[chunk_id];
   };

   //
   std::size_t find_bucket_idx(std::size_t size) {
      std::size_t min_bin = 0; // TODO: make impl more efficient curr = O(k)
      for (std::size_t i = 0; i < kBucketSizes.size(); ++i) {
         min_bin = kBucketSizes[i];
         if (size <= min_bin) {
            return i;
         }
      }
      return kBucketSizes.size() -
             1; // TODO: for below fallback behaviour we want to retry alloc
      // up in powers of 2
   };

   Bucket &find_bucket_for_ptr(void *ptr) {
      auto addr = reinterpret_cast<std::byte *>(ptr);
      for (auto &bucket : buckets_) {
         if (!bucket.has_mem_attatched)
            continue;
         auto base = reinterpret_cast<std::byte *>(bucket.ptr);
         auto end = base + bucket.region_size; // TODO: swap to end ptr
         if (addr >= base && addr < end) {
            return bucket;
         }
      }
      throw std::runtime_error("Failed to find bucket chunk ptr belongs too");
   };

   // TODO: This is the problematic code | chunk id is no longer just based on
   // bucket region
   ChunkId chunk_idx_for_ptr(Bucket &bucket, void *ptr) {
      auto base = reinterpret_cast<std::byte *>(bucket.ptr);
      auto addr = reinterpret_cast<std::byte *>(ptr);
      auto bucket_offset = bucket.bucket_id * kChunkCount;
      auto offset = addr - base;
      auto chunk_id = (offset / bucket.bucket_size) + bucket.first_chunk_idx;
      return chunk_id;
   }

   bool set_next_chunk(std::size_t idx, std::size_t chunk_count) {
      return idx + 1 <= chunk_count - 1;
   }
   bool set_prev_chunk(std::size_t idx) { return idx > 0; }

   void set_chunk_metadata(Bucket &bucket, Chunk &chunk, std::size_t size) {
      chunk.chunk_id = ChunkId{chunk_counter};
      if (set_next_chunk(chunk_counter, kChunkCount)) {
         chunk.next = ChunkId{chunk_counter + 1};
      }
      if (set_prev_chunk(chunk_counter)) {
         chunk.prev = ChunkId{chunk_counter - 1};
      }
      chunk.size = size;
      bucket.free_chunks.insert(chunk.chunk_id);
      chunks_.push_back(chunk);
      chunk.set_end_ptr();
   }

   void split_chunks(std::size_t bucket_index, std::size_t mem_size) {
      Bucket &bucket = buckets_[bucket_index];
      bucket.first_chunk_idx = chunk_counter;
      std::byte *byte_ptr = static_cast<std::byte *>(bucket.ptr);
      for (std::size_t i = 0; i < kChunkCount; ++i) {
         Chunk chunk;
         void *chunk_ptr =
             static_cast<void *>(byte_ptr + bucket.bucket_size * i);
         chunk.ptr = chunk_ptr;
         set_chunk_metadata(bucket, chunk, mem_size);
         chunk_counter++;
      }
   };

   void init_buckets() {
      for (std::size_t i = 0; i < kBucketSizes.size(); ++i) {
         buckets_[i] = Bucket();
         buckets_[i].bucket_id = i;
         buckets_[i].bucket_size = kBucketSizes[i];
      }
   };

   void *allocate_bucket_region(std::size_t region, std::size_t alignment) {
      void *ptr = sub_allocator_->allocate_region(alignment, region);
      return ptr;
   };
};

#endif // POOL_ALLOCATOR_H
