#ifndef BFC_POOL_H
#define BFC_POOL_H

#include <cstddef>
#include <cstdint>
#include <utility>

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
// NOLINTEND(misc-non-private-member-variables-in-classes)


struct Bucket {};

#endif // BFC_POOL_H
