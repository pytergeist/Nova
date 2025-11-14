#ifndef BFC_POOL_H
#define BFC_POOL_H

#include <cstddef>
#include <cstdint>
#include <utility>
#include <set>
#include "AllocatorInterface.h"

/* This header file includes code for the memory pool used by the pool
allocator, the pool consists of chunks, which point to mem regions - the
chunks store various meta-data, which describes whether a chunk is in use,
what the next/previous mem region is, the ptr to the curr mem region. */

using ChunkId = std::size_t;
static const ChunkId kInvalidChunkId = static_cast<ChunkId>(-1);

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)
struct Chunk {
   void *ptr = nullptr; // ptr to mem
   std::size_t size = 0;           // size of buffer
   std::size_t requested_size = 0; // the client requested size of the buffer
   std::int64_t allocation_id = -1;

   // next/prev allow iter to prev/next contiguous mem region
   ChunkId prev = kInvalidChunkId; // starts at ptr - prev->size
   ChunkId next = kInvalidChunkId; // starts at ptr + size

   bool in_use() const noexcept { return std::cmp_not_equal(allocation_id, kInvalidChunkId); }
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
   using FreeChunkSet = std::set<ChunkId, ChunkComparator>;
   std::size_t bucket_size = 0;
   FreeChunkSet free_chunks;

   Bucket(IAllocator* allocator, std::size_t bs) : bucket_size(bs), free_chunks(ChunkComparator(allocator)) {};
};
// NOLINTEND(misc-non-private-member-variables-in-classes)
#endif // BFC_POOL_H
