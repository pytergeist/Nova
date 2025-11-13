#ifndef POOL_H
#define POOL_H

#include <cstddef>
#include <cstdint>
#include <utility>

/* This header file includes code for the memory pool used by the pool
allocator, the pool consists of chunks, which point to mem regions - the
chunks store various meta-data, which describes whether a chunk is in use,
what the next/previous mem region is, the ptr to the curr mem region. */

using ChunkId = std::size_t;
static const ChunkId kInvalidChunkId = static_cast<ChunkId>(-1);

struct Chunk {
 public:

   bool in_use() const noexcept { return std::cmp_not_equal(allocation_id_, kInvalidChunkId); }

   std::size_t size() const noexcept { return size_; }
   std::size_t requested_size() const noexcept { return requested_size_; }
   std::int64_t allocation_id() const noexcept { return allocation_id_; }

   void set_size(std::size_t chunk_size) noexcept { size_ = chunk_size; }
   void set_requested_size(std::size_t req_chunk_size) noexcept {
      requested_size_ = req_chunk_size;
   }
   void set_allocation_id(std::int64_t alloc_id) noexcept {
      allocation_id_ = alloc_id;
   }

   ChunkId prev() const noexcept { return prev_; }
   ChunkId next() const noexcept { return next_; }

 private:
   void *ptr = nullptr; // ptr to mem
   std::size_t size_ = 0;           // size of buffer
   std::size_t requested_size_ = 0; // the client requested size of the buffer
   std::int64_t allocation_id_ = -1;

   // next/prev allow iter to prev/next contiguous mem region
   ChunkId prev_ = kInvalidChunkId; // starts at ptr - prev->size
   ChunkId next_ = kInvalidChunkId; // starts at ptr + size
};

#endif // POOL_H
