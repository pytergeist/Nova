#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H
#include <cstddef>
#include <iostream>

#include "AllocatorInterface.h"
#include "BFCPool.h"
#include "CPUSubAllocator.h"

static constexpr std::array<std::size_t, 5> kBucketSizes = {
    64, 128, 256, 512, 1024 //, 2048, 4096,
                            //    8192, 16384, 32768, 65536
};

static constexpr std::size_t kChunkCount = 64;

class PoolAllocator : public IAllocator {
 public:
   PoolAllocator() : sub_allocator_(std::make_unique<CPUSubAllocator>()) {
      init_buckets();
   };
   ~PoolAllocator() = default;

   void *allocate(std::size_t size, std::size_t alignment) override {
      // allocating a size/ alignment (stick with alignment = 64 for now) *
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
      if (chunk_ptr != nullptr) {
         return chunk_ptr;
      }
      throw std::bad_alloc();
   };

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
      bucket.chunks[id].in_use = false;
   };

   std::vector<Chunk> get_chunks_list(std::size_t bucket_size) {
      std::size_t bucket_index = find_bucket_idx(bucket_size);
      std::vector<Chunk> chunk_list = buckets_[bucket_index].chunks;
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

   std::size_t calc_region_size(std::size_t bucket_size,
                                std::size_t chunk_count) {
      return chunk_count * bucket_size;
   }

   Chunk &find_free_chunk(std::size_t bucket_index) {
      Bucket &bucket =
          buckets_[bucket_index]; // TODO: mechanism if nothing in free list?
      ChunkId chunk_id = *bucket.free_chunks.begin();
      bucket.free_chunks.erase(bucket.free_chunks.begin());
      return bucket.chunks[chunk_id];
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

   ChunkId chunk_idx_for_ptr(Bucket &bucket, void *ptr) {
      auto base = reinterpret_cast<std::byte *>(bucket.ptr);
      auto addr = reinterpret_cast<std::byte *>(ptr);
      auto offset = addr - base;
      return offset / bucket.bucket_size;
   }

   bool set_next_chunk(std::size_t idx, std::size_t chunk_count) {
      return idx + 1 <= chunk_count - 1;
   }
   bool set_prev_chunk(std::size_t idx) { return idx > 0; }

   void set_chunk_metadata(Bucket &bucket, Chunk &chunk, std::size_t id,
                           std::size_t size) {
      chunk.chunk_id = ChunkId{id};
      if (set_next_chunk(id, kChunkCount)) {
         chunk.next = ChunkId{id + 1};
      }
      if (set_prev_chunk(id)) {
         chunk.prev = ChunkId{id - 1};
      }
      chunk.size = size;
      bucket.free_chunks.insert(chunk.chunk_id);
      bucket.chunks.push_back(chunk);
      chunk.set_end_ptr();
   }

   void split_chunks(std::size_t bucket_index, std::size_t mem_size) {
      Bucket &bucket = buckets_[bucket_index];
      std::byte *byte_ptr = static_cast<std::byte *>(bucket.ptr);
      for (std::size_t i = 0; i < kChunkCount - 1; ++i) {
         Chunk chunk;
         void *chunk_ptr =
             static_cast<void *>(byte_ptr + bucket.bucket_size * i);
         chunk.ptr = chunk_ptr;
         set_chunk_metadata(bucket, chunk, i, mem_size);
      }
   };

   void init_buckets() {
      for (std::size_t i = 0; i < kBucketSizes.size(); ++i) {
         buckets_[i] = Bucket();
         buckets_[i].bucket_size = kBucketSizes[i];
      }
   };

   void *allocate_bucket_region(std::size_t region, std::size_t alignment) {
      void *ptr = sub_allocator_->allocate_region(alignment, region);
      return ptr;
   };
};

#endif // POOL_ALLOCATOR_H
