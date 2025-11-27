#ifndef BFC_POOL_H
#define BFC_POOL_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>

using ChunkId = std::size_t;
using BucketId = std::size_t;
static constexpr ChunkId kInvalidChunkId = static_cast<ChunkId>(-1);
static constexpr BucketId kInvalidBucketId = static_cast<BucketId>(-1);

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)
struct Chunk {
   void *ptr = nullptr; // ptr to mem sub region of chunk
   void *end_ptr_ = nullptr;
   std::size_t size = 0;           // size of buffer
   std::size_t requested_size = 0; // the client requested size of the buffer
   bool in_use = false;

   ChunkId chunk_id = kInvalidChunkId;
   // next/prev allow iter to prev/next contiguous mem region | CURRENTLY NOT
   // USED (for coalescing later) IMPORTANT NOTE: std::size_t = -1 becomes
   // SIZE_MAX(ALL BYTES = 1) = 18446744073709551615 (so dont be alarmed)
   ChunkId prev = kInvalidChunkId; // starts at ptr - prev->size
   ChunkId next = kInvalidChunkId; // starts at ptr + size

   void set_end_ptr() noexcept {
      assert(ptr != nullptr);
      assert(size != 0);
      std::byte *byte_end_ptr_ = static_cast<std::byte *>(ptr) + size;
      end_ptr_ = static_cast<void *>(byte_end_ptr_);
   }
};

struct ChunkComparator {
   const std::vector<Chunk> *chunks;

   bool operator()(ChunkId ca, ChunkId cb) const {
      const Chunk &a = (*chunks)[ca];
      const Chunk &b = (*chunks)[cb];
      if (a.size != b.size) {
         return a.size < b.size;
      }
      return a.ptr < b.ptr;
   }
};

struct Bucket {
   using FreeChunkSet = std::set<ChunkId, ChunkComparator>;
   BucketId bucket_id = 0;
   std::size_t bucket_size = 0;
   FreeChunkSet free_chunks;
   bool is_full() { return free_chunks.empty(); }

   Bucket(const std::vector<Chunk> *chunks, std::size_t bsize, std::size_t bid)
       : bucket_size(bsize), bucket_id(bid), free_chunks(ChunkComparator{chunks}) {}
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

#endif // BFC_POOL_H
