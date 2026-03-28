#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "Fusion/alloc/AllocTypes.h"
#include "Fusion/alloc/BFCPoolAllocator.h"
#include "Fusion/alloc/Pool.h"

#include "Fusion/common/Log.hpp"

class PoolAllocatorSplitMergeTest : public ::testing::Test {
 protected:
   PoolAllocator alloc;
};

static const Chunk *find_chunk_by_ptr(const PoolAllocator &alloc, void *ptr) {
   for (const auto &chunk : alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         return &chunk;
      }
   }
   return nullptr;
}

TEST_F(PoolAllocatorSplitMergeTest, AllocationSplitsChunkWhenRemainderIsLargeEnough) {
   void *ptr1 = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr1, nullptr);

   void *ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr2, nullptr);

   int live_chunk_count = 0;
   int free_chunk_count = 0;
   for (const Chunk &chunk : alloc.chunks()) {
      if (chunk.in_use) {
         live_chunk_count++;
      } else {
         free_chunk_count++;
      }
   }

   EXPECT_EQ(live_chunk_count, 2);
   EXPECT_EQ(free_chunk_count, 1);
}


TEST_F(PoolAllocatorSplitMergeTest, AllocationDoesNotSplitWhenRemainderTooSmall) {
   void *ptr1 = alloc.allocate(16, Alignment{64});
   ASSERT_NE(ptr1, nullptr);

   int nonzero_chunks = 0;
   for (const Chunk &chunk : alloc.chunks()) {
      if (chunk.size > 0) {
         nonzero_chunks++;
      }
   }

   EXPECT_EQ(nonzero_chunks, 1);
}


TEST_F(PoolAllocatorSplitMergeTest, FreeingChunkCanMergeWithNextFreeChunk) {
   void* p1 = alloc.allocate(128, Alignment{64});
   ASSERT_NE(p1, nullptr);

   void* p2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(p2, nullptr);

   alloc.deallocate(p2);

   bool found_free_128 = false;
   for (const auto& chunk : alloc.chunks()) {
      if (chunk.size == 128 && !chunk.in_use) {
         found_free_128 = true;
      }
   }

   EXPECT_TRUE(found_free_128);
}

TEST_F(PoolAllocatorSplitMergeTest, FreeingChunkCanMergeWithPreviousFreeChunk) {
   void *ptr1 = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr1, nullptr);

   void *ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr2, nullptr);

   void* ptr3 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr3, nullptr);

   alloc.deallocate(ptr2);
   alloc.deallocate(ptr3);

   bool found_free_128 = false;
   int free_64_count = 0;
   for (const Chunk &chunk : alloc.chunks()) {
      if (chunk.size == 0) {
         continue;
      }
      if (chunk.size == 128 && !chunk.in_use) {
         found_free_128 = true;
      }
      if (chunk.size == 64 && !chunk.in_use) {
         free_64_count++;
      }
   }
   EXPECT_TRUE(found_free_128);
   EXPECT_EQ(free_64_count, 0);
}


TEST_F(PoolAllocatorSplitMergeTest, PreviousMergedChunkAppearsInLargerBucket) {
   void *ptr1 = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr1, nullptr);

   void *ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr2, nullptr);

   void* ptr3 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr3, nullptr);

   alloc.deallocate(ptr2);
   alloc.deallocate(ptr3);

   std::vector<ChunkID> free_chunks_128 = alloc.get_free_chunks(128);
   std::vector<ChunkID> free_chunks_64 = alloc.get_free_chunks(64);
   EXPECT_FALSE(free_chunks_128.empty());
   EXPECT_TRUE(free_chunks_64.empty());
}


TEST_F(PoolAllocatorSplitMergeTest, NextMergedChunkAppearsInLargerBucket) {
   void *ptr1 = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr1, nullptr);

   void *ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr2, nullptr);

   alloc.deallocate(ptr2);

   std::vector<ChunkID> free_chunks_128 = alloc.get_free_chunks(128);
   std::vector<ChunkID> free_chunks_64 = alloc.get_free_chunks(64);
   EXPECT_FALSE(free_chunks_128.empty());
   EXPECT_TRUE(free_chunks_64.empty());
}


TEST_F(PoolAllocatorSplitMergeTest, AdjacentFreedChunksShouldCoalesceRegardlessOfFreeOrder) {
   void *ptr1 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr1, nullptr);
   void *ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr2, nullptr);

   alloc.deallocate(ptr2);
   alloc.deallocate(ptr1);

   int total_free = 0;
   int nonzero_free_chunks = 0;

   for (const Chunk &chunk : alloc.chunks()) {
      if (chunk.size == 0) {
         continue;
      }
      if (!chunk.in_use) {
         total_free += chunk.size;
         nonzero_free_chunks++;
      }
   }
	EXPECT_EQ(nonzero_free_chunks, 1);
	EXPECT_GE(total_free, 128);
}


