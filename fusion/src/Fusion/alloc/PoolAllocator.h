#ifndef POOL_ALLOCATOR_H
#define POOL_ALLOCATOR_H

#include <cstddef>
#include <iostream>
#include "BFCPool.h"
#include "AllocatorInterface.h"
#include "CPUSubAllocator.h"

static constexpr std::array<std::size_t, 5> kBucketSizes = {
    64, 128, 256, 512, 1024//, 2048, 4096,
//    8192, 16384, 32768, 65536
};

static constexpr std::size_t kChunkCount = 64;


class PoolAllocator : public IAllocator {
   public:

     PoolAllocator() {init_buckets();};
     ~PoolAllocator() = default;

     void* allocate(std::size_t size, std::size_t alignment) override {
        // allocating a size/ alignment (stick with alignment = 64 for now) *
		std::size_t bucket_index = find_bucket_idx(size);
        std::cout << kBucketSizes[bucket_index] << std::endl;
		if (!buckets_[bucket_index].has_mem_attatched) {
            std::size_t region = calc_region_size(kBucketSizes[bucket_index], kChunkCount);
        	void* ptr = allocate_bucket_region(region, alignment);
            buckets_[bucket_index].ptr = ptr;
            buckets_[bucket_index].region_size = region;
            buckets_[bucket_index].has_mem_attatched = true;
            split_chunks(bucket_index, size, region / kChunkCount);
		}
        Chunk& chunk = find_free_chunk(bucket_index);
        void* chunk_ptr = chunk.ptr;
        // TODO: currently no mechanism for grabbing existing bucket
        // TODO: map ptr back to chunk?
        // set chunk metadata (ptr bumping? set prev/next)
        return chunk_ptr;
     };

     Chunk& find_free_chunk(std::size_t bucket_index) {
        Bucket& bucket = buckets_[bucket_index]; // TODO: mechanism if nothing in free list?
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
        return kBucketSizes.size() - 1; // TODO: for below fallback behaviour we want to retry alloc
        // up in powers of 2
     };

     void deallocate(void* p) override {};


     bool set_next_chunk(std::size_t idx, std::size_t chunk_count) {
        return idx + 1 <= chunk_count - 1;
     }
     bool set_prev_chunk(std::size_t idx) {
        return idx - 1 > 0;
     }

     void set_chunk_metadata(Bucket& bucket, Chunk& chunk, std::size_t id, std::size_t req_size, std::size_t mem_size) {
       chunk.chunk_id = ChunkId{id};
       if (set_next_chunk(id, kChunkCount)) {
        chunk.next = ChunkId{id + 1};
       }
       if (set_prev_chunk(id)) {
        chunk.prev = ChunkId{id - 1};
       }
       bucket.free_chunks.insert(chunk.chunk_id);
       bucket.chunks.push_back(chunk);
	   chunk.size = mem_size;
       chunk.requested_size = req_size;
       chunk.set_end_ptr();
     }

     void split_chunks(std::size_t bucket_index, std::size_t req_size, std::size_t mem_size) {
		Bucket& bucket = buckets_[bucket_index];
        std::byte* byte_ptr = static_cast<std::byte*>(bucket.ptr);
        for (std::size_t i = 0; i < kChunkCount - 1; ++i) {
           Chunk chunk;
           void* chunk_ptr = static_cast<void*>(byte_ptr + bucket.bucket_size * i);
           chunk.ptr = chunk_ptr;
           set_chunk_metadata(bucket, chunk, i, req_size, mem_size);
        }
     };

     const Chunk* ChunkFromId(const ChunkId cid) override {
        return nullptr;
     };

   private:
    CPUSubAllocator sub_allocator_;
    std::array<Bucket, kBucketSizes.size()> buckets_;

    std::size_t calc_region_size(std::size_t bucket_size, std::size_t chunk_count) {
       return chunk_count * bucket_size;
    }
    void init_buckets() {
       for (std::size_t i = 0; i < kBucketSizes.size(); ++i) {
          buckets_[i] = Bucket();
          buckets_[i].bucket_size = kBucketSizes[i];
       }
    };

    void* allocate_bucket_region(std::size_t region, std::size_t alignment) {
       void* ptr = sub_allocator_.allocate_region(alignment, region);
       return ptr;
    };

};

#endif // POOL_ALLOCATOR_H
