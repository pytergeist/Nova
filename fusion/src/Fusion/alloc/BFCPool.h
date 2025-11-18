#ifndef BFC_POOL_H
#define BFC_POOL_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>

#include "AllocatorInterface.h"

/* This header file includes code for the memory pool used by the pool
allocator, the pool consists of chunks, which point to mem regions - the
chunks store various meta-data, which describes whether a chunk is in use,
what the next/previous mem region is, the ptr to the curr mem region. */

using ChunkId = std::size_t;
static const ChunkId kInvalidChunkId = static_cast<ChunkId>(-1);

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

// Add back in later - this is for size ordering
// struct ChunkComparator {
//   public:
//     explicit ChunkComparator(IAllocator* allocator) : allocator_(allocator)
//     {}; bool operator()(const ChunkId& ca, const ChunkId& cb) const {
//        const Chunk* a = allocator_->ChunkFromId(ca); // TODO: impl
//        ChunkFromId const Chunk* b = allocator_->ChunkFromId(cb); if (a->size
//        != b->size) {
//           return a->size < b->size;
//        }
//        return a->ptr < b->ptr;
//     }
//   private:
//     IAllocator* allocator_;
//};

struct Bucket {
   using FreeChunkSet = std::set<ChunkId>; //, ChunkComparator>;
   void *ptr = nullptr;                    // base region ptr (used for freeing)
   // below does not have proper ordering yet - once coalescing in impl - will
   // need a customer comparator
   std::size_t bucket_id = 0;
   std::size_t bucket_size = 0;
   std::size_t region_size = 0;
   std::size_t first_chunk_idx = -1;
   FreeChunkSet free_chunks;
   bool has_mem_attatched = false;

   Bucket() : bucket_size(0), region_size(0), has_mem_attatched(false) {};
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

#endif // BFC_POOL_H
