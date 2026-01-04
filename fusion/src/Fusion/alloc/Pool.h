// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef BFC_POOL_H
#define BFC_POOL_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)
struct ChunkID {
   std::size_t value;
   operator std::size_t() const noexcept { return value; }
};
struct BucketID {
   std::size_t value;
   operator std::size_t() const noexcept { return value; }
};

static constexpr ChunkID kInvalidChunkID = static_cast<ChunkID>(-1);
static constexpr BucketID kInvalidBucketID = static_cast<BucketID>(-1);

struct Chunk {
   void *ptr = nullptr; // ptr to mem sub region of chunk
   void *end_ptr_ = nullptr;
   std::size_t size = 0;           // size of buffer
   std::size_t requested_size = 0; // the client requested size of the buffer
   bool in_use = false;

   ChunkID chunk_id = kInvalidChunkID;
   // next/prev allow iter to prev/next contiguous mem region | CURRENTLY NOT
   // USED (for coalescing later) IMPORTANT NOTE: std::size_t = -1 becomes
   // SIZE_MAX(ALL BYTES = 1) = 18446744073709551615 (so dont be alarmed)
   ChunkID prev = kInvalidChunkID; // starts at ptr - prev->size
   ChunkID next = kInvalidChunkID; // starts at ptr + size

   void set_end_ptr() noexcept {
      assert(ptr != nullptr);
      assert(size != 0);
      std::byte *byte_end_ptr_ = static_cast<std::byte *>(ptr) + size; // NOLINT
      end_ptr_ = static_cast<void *>(byte_end_ptr_);
   }
};

struct ChunkComparator {
   const std::vector<Chunk> *chunks;

   bool operator()(ChunkID ca, ChunkID cb) const {
      const Chunk &a = (*chunks).at(ca);
      const Chunk &b = (*chunks).at(cb);
      if (a.size != b.size) {
         return a.size < b.size;
      }
      return a.ptr < b.ptr;
   }
};

struct Bucket {
   using FreeChunkSet = std::set<ChunkID, ChunkComparator>;
   BucketID bucket_id = kInvalidBucketID;
   std::size_t bucket_size = 0;
   FreeChunkSet free_chunks;
   bool is_full() const { return free_chunks.empty(); }

   Bucket(const std::vector<Chunk> *chunks, std::size_t bsize, BucketID bid)
       : bucket_size(bsize), bucket_id(bid),
         free_chunks(ChunkComparator{chunks}) {}
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

#endif // BFC_POOL_H
