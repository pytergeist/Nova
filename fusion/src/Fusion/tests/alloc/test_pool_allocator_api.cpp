#include <gtest/gtest.h>
#include <vector>
#include <cstddef>

#include "Fusion/alloc/AllocTypes.h"
#include "Fusion/alloc/Pool.h"
#include "Fusion/alloc/BFCPoolAllocator.h"

class PoolAllocatorTest : public ::testing::Test {
protected:
   PoolAllocator alloc;
};

TEST_F(PoolAllocatorTest, AllocateReturnsNonNull) {
   void* ptr = alloc.allocate(64, Alignment{64});
   EXPECT_NE(ptr, nullptr);
}


TEST_F(PoolAllocatorTest, ZeroSizeAllocationSucceeds) {
   void* ptr = alloc.allocate(0, Alignment{64});
   EXPECT_NE(ptr, nullptr);
}

TEST_F(PoolAllocatorTest, DeallocateNullptrDoesNothing) {
   EXPECT_NO_THROW(alloc.deallocate(nullptr));
}

TEST_F(PoolAllocatorTest, AllocatedChunkIsMarkedInUse) {
   void* ptr = alloc.allocate(64, Alignment{64});
   EXPECT_NE(ptr, nullptr);

   bool found = false;
   for (Chunk& chunk: alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         found = true;
         EXPECT_TRUE(chunk.in_use);
         EXPECT_EQ(chunk.requested_size, 64);
      }
    }
   EXPECT_TRUE(found);
}

TEST_F(PoolAllocatorTest, ExactPowerOfTwoRequestKeepsRequestedSize) {
   void* ptr = alloc.allocate(128, Alignment{64});
   EXPECT_NE(ptr, nullptr);

   bool found = false;
   for (Chunk& chunk: alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         found = true;
         EXPECT_TRUE(chunk.in_use);
         EXPECT_EQ(chunk.requested_size, 128);
         EXPECT_EQ(chunk.requested_size, 128);
      }
   }
   EXPECT_TRUE(found);
}

TEST_F(PoolAllocatorTest, NonPowerOfTwoRequestRoundsUp) {
   void* ptr = alloc.allocate(100, Alignment{64});
   EXPECT_NE(ptr, nullptr);

   bool found = false;
   for (Chunk& chunk: alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         found = true;
         EXPECT_TRUE(chunk.in_use);
         EXPECT_EQ(chunk.requested_size, 128);
         EXPECT_EQ(chunk.requested_size, 128);
      }
   }
   EXPECT_TRUE(found);
}


TEST_F(PoolAllocatorTest, DeallocateMarksChunkAsFree) {
   void* ptr = alloc.allocate(100, Alignment{64});
   EXPECT_NE(ptr, nullptr);

   alloc.deallocate(ptr);

   bool found = false;
   for (Chunk& chunk: alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         found = true;
         EXPECT_FALSE(chunk.in_use);
         EXPECT_EQ(chunk.requested_size, 0);
      }
   }
   EXPECT_TRUE(found);
}


TEST_F(PoolAllocatorTest, ReusesFreedChunkForSameSizeAllocation) {
   void* ptr1 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr1, nullptr);

   alloc.deallocate(ptr1);

   void* ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr2, nullptr);
   ASSERT_EQ(ptr1, ptr2);
}

TEST_F(PoolAllocatorTest, DeallocatingForeignPointerThrows) {
   int x = 0;
   EXPECT_THROW(alloc.deallocate(&x), std::runtime_error);
}

TEST_F(PoolAllocatorTest, DoubleFreeThrows) {
   void* ptr = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   alloc.deallocate(ptr);

   EXPECT_THROW(alloc.deallocate(ptr), std::runtime_error);
}
