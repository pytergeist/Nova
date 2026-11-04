#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "Fusion/core/memory/alloc/FUAllocator.h"

class FUAllocatorFuzzTest : public ::testing::Test {
 protected:
   FUAllocator alloc;
};

static bool overlaps(const Chunk &a, const Chunk &b) {
   std::byte *a_begin = static_cast<std::byte *>(a.ptr);
   std::byte *a_end = a_begin + a.size;
   std::byte *b_begin = static_cast<std::byte *>(b.ptr);
   std::byte *b_end = b_begin + b.size;

   return (a_begin < b_end) && (b_begin < a_end);
}

static void validate_allocator_invariants(const FUAllocator &alloc) {
   const std::vector<Chunk> chunks = alloc.chunks();

   for (const Chunk &chunk : chunks) {
      if (chunk.size == 0) {
         EXPECT_EQ(chunk.ptr, nullptr);
         EXPECT_EQ(chunk.requested_size, 0U);
         EXPECT_FALSE(chunk.in_use);
         continue;
      }

      EXPECT_NE(chunk.ptr, nullptr);

      if (chunk.in_use) {
         EXPECT_GT(chunk.requested_size, 0U);
      } else {
         EXPECT_EQ(chunk.requested_size, 0U);
      }

      if (chunk.prev != kInvalidChunkID) {
         ASSERT_LT(static_cast<std::size_t>(chunk.prev), chunks.size());
         const Chunk &prev = chunks[chunk.prev];
         EXPECT_EQ(prev.next, chunk.chunk_id);
      }

      if (chunk.next != kInvalidChunkID) {
         ASSERT_LT(static_cast<std::size_t>(chunk.next), chunks.size());
         const Chunk &next = chunks[chunk.next];
         EXPECT_EQ(next.prev, chunk.chunk_id);
      }
   }

   std::vector<const Chunk *> live_chunks;
   for (const Chunk &chunk : chunks) {
      if (chunk.size > 0 && chunk.in_use) {
         live_chunks.push_back(&chunk);
      }
   }

   for (std::size_t i = 0; i < live_chunks.size(); ++i) {
      for (std::size_t j = i + 1; j < live_chunks.size(); ++j) {
         EXPECT_FALSE(overlaps(*live_chunks[i], *live_chunks[j]));
      }
   }

   for (std::size_t bucket_size : {1UL, 2UL, 4UL, 8UL, 16UL, 32UL, 64UL, 128UL,
                                   256UL, 512UL, 1024UL, 2048UL, 4096UL}) {
      const std::vector<ChunkID> chunk_ids = alloc.get_free_chunks(bucket_size);
      for (ChunkID chunk_id : chunk_ids) {
         ASSERT_LT(static_cast<std::size_t>(chunk_id), chunks.size());
         const Chunk &chunk = chunks[chunk_id];
         EXPECT_GT(chunk.size, 0U);
         EXPECT_FALSE(chunk.in_use);
      }
   }
}

TEST_F(FUAllocatorFuzzTest,
       random_allocate_free_seqwuence_maintains_invariants) {
   std::mt19937 rng(420420);
   std::uniform_int_distribution<int> coin(0, 1);
   std::uniform_int_distribution<int> size_pick(0, 9);

   const std::vector<std::size_t> sizes = {1,  2,  3,  8,   16,
                                           24, 64, 70, 128, 256};

   std::vector<void *> live_ptrs;

   for (int step = 0; step < 1000; ++step) {
      const bool should_allocate = live_ptrs.empty() || coin(rng) == 0;

      if (should_allocate) {
         const std::size_t size = sizes[size_pick(rng)];
         void *ptr = alloc.allocate(size, Alignment{64});
         ASSERT_NE(ptr, nullptr);
         live_ptrs.push_back(ptr);
      } else {
         std::uniform_int_distribution<std::size_t> index_pick(
             0, live_ptrs.size() - 1);
         const std::size_t idx = index_pick(rng);

         void *ptr = live_ptrs[idx];
         alloc.deallocate(ptr);
         live_ptrs.erase(live_ptrs.begin() + static_cast<std::ptrdiff_t>(idx));
      }

      validate_allocator_invariants(alloc);
   }

   for (void *ptr : live_ptrs) {
      alloc.deallocate(ptr);
      validate_allocator_invariants(alloc);
   }
}

TEST_F(FUAllocatorFuzzTest, random_free_order_maintains_invariants) {
   std::mt19937 rng(420420);
   std::uniform_int_distribution<int> size_pick(0, 6);

   const std::vector<std::size_t> sizes = {
       8,    16,   32,   64,    128,   256,   512,    1024,
       2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

   std::vector<void *> ptrs;

   for (int i = 0; i < 200; ++i) {
      void *ptr = alloc.allocate(sizes[size_pick(rng)], Alignment{64});
      ASSERT_NE(ptr, nullptr);
      ptrs.push_back(ptr);
   }

   validate_allocator_invariants(alloc);

   std::shuffle(ptrs.begin(), ptrs.end(), rng);

   for (void *ptr : ptrs) {
      alloc.deallocate(ptr);
      validate_allocator_invariants(alloc);
   }
}

TEST_F(FUAllocatorFuzzTest,
       repeated_same_size_allocate_free_maintains_invariants) {
   for (int i = 0; i < 500; ++i) {
      void *ptr = alloc.allocate(64, Alignment{64});
      ASSERT_NE(ptr, nullptr);
      validate_allocator_invariants(alloc);

      alloc.deallocate(ptr);
      validate_allocator_invariants(alloc);
   }
}