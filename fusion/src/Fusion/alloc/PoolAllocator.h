#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H

#include <cstddef>
#include <iostream>

#include "../common/Checks.h"
#include "../common/Log.h"
#include "AllocatorInterface.h"
#include "BFCPool.h"
#include "CPUSubAllocator.h"

static constexpr std::size_t kMinAllocationSize = 64;
static constexpr std::size_t kNumBuckets = 30;
static constexpr std::size_t kMinSplitFactor = 2;

constexpr std::array<std::size_t, kNumBuckets> make_bucket_sizes() {
   std::array<std::size_t, kNumBuckets> sizes{};
   std::size_t value = kMinAllocationSize;
   for (std::size_t i = 0; i < kNumBuckets; ++i) {
      sizes[i] = value;
      value <<= 1;
   }
   return sizes;
}

static constexpr std::array<std::size_t, kNumBuckets> kBucketSizes =
    make_bucket_sizes();

static constexpr std::size_t kChunkCount = 64;

struct Region {
   void *ptr;
   std::size_t region_id;
   std::size_t size;
   std::size_t alignment;
};

class RegionManager {
   /* The region manager is responsible for two main parts of the allocation
    * process. It tracks metadata about region size (these regions are produced
    * by the suballocator), this meta data include ptr (to the beginnign of the
    * alloc region), the total size of the region and the regions alignment.
    * Provides a mechanism for pointer lookup, for identifying the region a ptr
    * exists in and the specific chunkID of that region that the ptr belongs to.
    *    */
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
         auto end = base + region.size; // TODO: swap to end ptr
         if (addr >= base && addr < end) {
            return region;
         }
      }
      throw std::runtime_error("Failed to find bucket chunk ptr belongs too");
   };

   ChunkId get_chunkid_from_ptr(void *chunk_ptr) {
      auto it = ptr_chunk_map_.find(chunk_ptr);
      if (it == ptr_chunk_map_.end()) {
         FUSION_LOGI(false,
                     "PoolAllocator: deallocate called with unknown pointer: ",
                     chunk_ptr, " (double free or foreign pointer)");
         throw std::runtime_error("Failed to allocate a chunk");
      }
      return it->second;
   }

   void set_chunkid(void *chunk_ptr, std::size_t chunk_id) {
      // NB: Insert or assign needed as we're reusing ptr for different chunkIds
      // after coalesce
      ptr_chunk_map_.insert_or_assign(chunk_ptr, chunk_id);
   }

   bool erase_chunk(void *chunk_ptr) {
      bool success = ptr_chunk_map_.erase(chunk_ptr);
      // returns 1 if key exists, 0 if not
      return success;
   }

   std::vector<Region> regions() { return regions_; };
   std::vector<Region> regions() const { return regions_; };

 private:
   std::unordered_map<void *, ChunkId> ptr_chunk_map_;
   std::vector<Region> regions_;
   std::size_t counter_ = 0;
};

class PoolAllocator : public IAllocator {
 public:
   PoolAllocator() : sub_allocator_(std::make_unique<CPUSubAllocator>()) {
      init_buckets();
   };
   ~PoolAllocator() = default;
   std::size_t chunk_counter = 0;

   std::size_t round_bytes(std::size_t n) {
      if (n <= 1)
         return 1;

      int bw = std::bit_width(n - 1);
      if (bw >= int(sizeof(std::size_t) * 8))
         return std::numeric_limits<std::size_t>::max(); // clamp

      return std::size_t{1} << bw;
   }

   void *allocate(std::size_t size, std::size_t alignment) override {
      if (size > kBucketSizes.back()) {
         FUSION_LOGI("Trying to allocate size: ", size,
                     " with largest bucket size ", kBucketSizes.back());
         throw std::bad_alloc();
      };
      size = round_bytes(size);
      Bucket &bucket = find_free_bucket(size);
      if (size > kBucketSizes.back()) {
         FUSION_LOGI("Requested rounded size: ", size);
         FUSION_LOGI("Max Bucket size: ", kBucketSizes.back());
         throw std::runtime_error("Trying to allocate more than one bucket");
      }
      if (!bucket.has_mem_attatched) {
         std::size_t region = calc_region_size(bucket.bucket_size, kChunkCount);
         void *ptr = allocate_bucket_region(region, alignment);
         bucket.ptr = ptr;
         bucket.has_mem_attatched = true;
         split_chunks(bucket, region / kChunkCount);
      }
      Chunk &chunk = find_free_chunk(bucket);

      std::size_t split_factor = chunk.size / size;
      if (split_factor > kMinSplitFactor) {
         std::size_t new_bucket_size = chunk.size / split_factor;
         std::size_t bucket_idx = find_bucket_idx(new_bucket_size);
         Bucket &new_bucket = buckets_[bucket_idx];
         chunk = tmp_split(new_bucket, chunk, split_factor);
      }

      void *chunk_ptr = chunk.ptr;
      chunk.in_use = true;
      chunk.requested_size = size;
      //      FUSION_CHECK(size < chunk.size, "Trying to allocate mem size >
      //      chunk.size");
      if (chunk_ptr != nullptr) {
         return chunk_ptr;
      }
      throw std::bad_alloc();
   };

   void reset_chunk_metadata(Chunk &chunk) {
      chunk.in_use = false;
      chunk.requested_size = 0;
   }

   Chunk &get_chunk_from_id(ChunkId chunk_id) {
      FUSION_BOUNDS_CHECK(chunk_id, chunks_.size());
      return chunks_[chunk_id];
   }

   void delete_chunk(Chunk &chunk) {
      chunk.size = 0;
      chunk.requested_size = 0;
      chunk.ptr = nullptr;
      chunk.end_ptr_ = nullptr;
      chunk.prev = kInvalidChunkId;
      chunk.next = kInvalidChunkId;
   }

   void erase_chunk_from_bucket(Chunk &chunk) {
      Bucket &bucket = buckets_.at(find_bucket_idx(chunk.size));
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

         // set next for merged chunk to be the next of right chunk
         ChunkId rnext_id = rchunk.next;
         lchunk.next = rnext_id;
         // here we set the prev of the new next chunk
         if (rnext_id != kInvalidChunkId) {
            Chunk &rnext = get_chunk_from_id(rnext_id);
            rnext.prev = lchunk.chunk_id;
         }
         delete_chunk(rchunk);
         return lchunk.chunk_id;
      }
      return rchunk.chunk_id;
   }

   // TODO: use a while loop you dummy
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

   void deallocate(void *ptr) override {
      if (!ptr)
         return;
      ChunkId chunk_id = region_manager_.get_chunkid_from_ptr(ptr);
      Chunk &chunk = get_chunk_from_id(chunk_id);

      //      FUSION_CHECK(chunk.in_use,
      //                   "Trying to deallocate chunk not currently in use!");

      chunk_id = free_and_maybe_coalesce(chunk_id);
      //        if (!bucket) {
      //           // TODO: err handle?
      //           return;
      //        }
      //	  FUSION_LOGI("Deallocating chunk: ", chunk.chunk_id, ", with
      // ptr= ", chunk.ptr);
      std::size_t bucket_index =
          find_bucket_idx(round_bytes(chunks_[chunk_id].size));
      Bucket &bucket = buckets_[bucket_index];
      bucket.free_chunks.insert(chunk_id);
      reset_chunk_metadata(chunks_[chunk_id]);
   };

   std::vector<Chunk> chunks() { return chunks_; }

   std::set<ChunkId> get_free_chunks(std::size_t bucket_size) {
      std::size_t bucket_index = find_bucket_idx(bucket_size);
      std::set<ChunkId> free_chunks_set = buckets_[bucket_index].free_chunks;
      return free_chunks_set;
   }

 private:
   std::unique_ptr<ISubAllocator> sub_allocator_;
   std::array<Bucket, kBucketSizes.size()> buckets_;
   std::vector<Chunk> chunks_;
   RegionManager region_manager_;

   std::size_t calc_region_size(std::size_t bucket_size,
                                std::size_t chunk_count) {
      return chunk_count * bucket_size;
   }

   Chunk &find_free_chunk(Bucket &bucket) {
      auto it = bucket.free_chunks.begin();
      if (it != bucket.free_chunks.end()) {
         ChunkId chunk_id = *it;
         bucket.free_chunks.erase(chunk_id);
         return get_chunk_from_id(chunk_id);
      } else {
         throw std::runtime_error("No free chunk found!");
      }
   };

   // std::size_t find_bucket_idx(std::size_t size) {
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

   Bucket &find_free_bucket(std::size_t size) {
      std::size_t min_bin = 0; // TODO: make impl more efficient curr = O(k)
      for (auto &bucket : buckets_) {
         if (size <= bucket.bucket_size && !bucket.is_full()) {
            return bucket;
         }
      }
      throw std::runtime_error(
          "No free buckets left"); // TODO: for below fallback behaviour we want
                                   // to retry alloc
      // up in powers of 2
   };

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
      chunks_.push_back(chunk); // TODO: change to emplace_back?
      chunk.set_end_ptr();
   }

   void allocate_chunk(void *chunk_ptr, ChunkId ochunk_id, std::size_t size,
                       std::size_t idx) {
      // TODO: the prev/next logic here makes no sense
      Chunk chunk;
      chunk.ptr = chunk_ptr;
      chunk.chunk_id = chunk_counter;
      chunk.prev = ChunkId{ochunk_id + idx};
      chunk.next = ChunkId{ochunk_id + idx + 1};
      chunk.size = size;
      chunk.set_end_ptr();
      region_manager_.set_chunkid(chunk_ptr, chunk.chunk_id);
      chunks_.push_back(chunk);
      chunk_counter++;
   }

   Chunk &tmp_split(Bucket &bucket, Chunk &ochunk, std::size_t split_factor) {
      std::size_t osize = ochunk.size;
      std::size_t nsize = osize / split_factor;
      std::size_t ochunk_id = chunk_counter;
      std::byte *byte_ptr = reinterpret_cast<std::byte *>(ochunk.ptr);
      for (std::size_t i = 0; i < split_factor; ++i) {
         void *chunk_ptr = reinterpret_cast<void *>(byte_ptr + i * nsize);
         allocate_chunk(chunk_ptr, ochunk_id, nsize, i);
      }
      return get_chunk_from_id(ochunk_id);
   }

   //      std::size_t first_chunk_idx = chunk_counter;
   //      std::byte *byte_ptr = reinterpret_cast<std::byte *>(ochunk.ptr);
   //      for (std::size_t i = 0; i < split_factor; ++i) {
   //         Chunk chunk;
   //         void *chunk_ptr =
   //             reinterpret_cast<void *>(byte_ptr + bucket.bucket_size * i);
   //         chunk.ptr = chunk_ptr;
   //         set_chunk_metadata(bucket, chunk, bucket.bucket_size);
   //         region_manager_.set_chunkid(chunk_ptr, chunk.chunk_id);
   //         chunk_counter++;
   //      }
   //      return get_chunk_from_id(first_chunk_idx);
   //   }

   void split_chunks(Bucket &bucket, std::size_t mem_size) {
      std::byte *byte_ptr = static_cast<std::byte *>(bucket.ptr);
      for (std::size_t i = 0; i < kChunkCount; ++i) {
         Chunk chunk;
         void *chunk_ptr =
             static_cast<void *>(byte_ptr + bucket.bucket_size * i);
         chunk.ptr = chunk_ptr;
         set_chunk_metadata(bucket, chunk, mem_size);
         region_manager_.set_chunkid(chunk_ptr, chunk.chunk_id);
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
      region_manager_.add_allocated_region(ptr, region, alignment);
      return ptr;
   };
};

#endif // POOL_ALLOCATOR_H
