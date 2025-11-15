#ifndef BFC_POOL_H
#define BFC_POOL_H

#include <cstddef>
#include <cstdint>
#include <utility>
#include <set>
#include <cassert>
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
   void* end_ptr_ = nullptr;
   std::size_t size = 0;           // size of buffer
   std::size_t requested_size = 0; // the client requested size of the buffer

   ChunkId chunk_id = kInvalidChunkId;
   // next/prev allow iter to prev/next contiguous mem region
   ChunkId prev = kInvalidChunkId; // starts at ptr - prev->size
   ChunkId next = kInvalidChunkId; // starts at ptr + size

   bool in_use() const noexcept { return std::cmp_not_equal(chunk_id, kInvalidChunkId); }
   void set_end_ptr() noexcept {
      assert(ptr != nullptr);
      assert(size != 0);
      std::byte* byte_end_ptr_ = static_cast<std::byte*>(ptr) + size;
      end_ptr_ = static_cast<void*>(byte_end_ptr_);
   }
};

struct ChunkComparator {
   public:
     explicit ChunkComparator(IAllocator* allocator) : allocator_(allocator) {};
     bool operator()(const ChunkId& ca, const ChunkId& cb) const {
        const Chunk* a = allocator_->ChunkFromId(ca); // TODO: impl ChunkFromId
        const Chunk* b = allocator_->ChunkFromId(cb);
        if (a->size != b->size) {
           return a->size < b->size;
        }
        return a->ptr < b->ptr;
     }
   private:
     IAllocator* allocator_;
};


struct Bucket {
   void* ptr = nullptr; // base region ptr (used for freeing)
   using FreeChunkSet = std::set<ChunkId>; //, ChunkComparator>;
   std::vector<Chunk> chunks;
   std::size_t bucket_size = 0;
   std::size_t region_size = 0;
   FreeChunkSet free_chunks;
   bool has_mem_attatched = false;

   Bucket(): bucket_size(0), region_size(0), has_mem_attatched(false) {};
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

#endif // BFC_POOL_H
