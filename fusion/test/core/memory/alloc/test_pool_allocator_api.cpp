#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "Fusion/core/memory/alloc/AllocTypes.h"
#include "Fusion/core/memory/alloc/FUAllocator.h"
#include "Fusion/core/memory/alloc/Pool.h"

class FUAllocatorTest : public ::testing::Test {
 protected:
   FUAllocator alloc;
};

TEST_F(FUAllocatorTest, allocation_returns_non_null) {
   void *ptr = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr, nullptr);
}

TEST_F(FUAllocatorTest, zero_size_allocation_succeeds) {
   void *ptr = alloc.allocate(0, Alignment{64});
   ASSERT_NE(ptr, nullptr);
}

TEST_F(FUAllocatorTest, deallocation_null_ptr_does_nothing) {
   EXPECT_NO_THROW(alloc.deallocate(nullptr));
}

TEST_F(FUAllocatorTest, allocated_chunk_is_marked_in_use) {
   void *ptr = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   bool found = false;
   for (Chunk &chunk : alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         found = true;
         EXPECT_TRUE(chunk.in_use);
         EXPECT_EQ(chunk.requested_size, 64);
      }
   }
   EXPECT_TRUE(found);
}

TEST_F(FUAllocatorTest, exact_power_of_two_request_keeps_requested_size) {
   void *ptr = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   bool found = false;
   for (Chunk &chunk : alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         found = true;
         EXPECT_TRUE(chunk.in_use);
         EXPECT_EQ(chunk.requested_size, 128);
         EXPECT_EQ(chunk.requested_size, 128);
      }
   }
   EXPECT_TRUE(found);
}

TEST_F(FUAllocatorTest, non_power_of_two_request_rounds_up) {
   void *ptr = alloc.allocate(100, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   bool found = false;
   for (Chunk &chunk : alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         found = true;
         EXPECT_TRUE(chunk.in_use);
         EXPECT_EQ(chunk.requested_size, 128);
         EXPECT_EQ(chunk.requested_size, 128);
      }
   }
   EXPECT_TRUE(found);
}

TEST_F(FUAllocatorTest, deallocate_marks_chunks_as_free) {
   void *ptr = alloc.allocate(100, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   alloc.deallocate(ptr);

   bool found = false;
   for (Chunk &chunk : alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         found = true;
         EXPECT_FALSE(chunk.in_use);
         EXPECT_EQ(chunk.requested_size, 0);
      }
   }
   EXPECT_TRUE(found);
}

TEST_F(FUAllocatorTest, reuses_freed_chunk_for_same_size_allocation) {
   void *ptr1 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr1, nullptr);

   alloc.deallocate(ptr1);

   void *ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr2, nullptr);
   EXPECT_EQ(ptr1, ptr2);
}

TEST_F(FUAllocatorTest, deallocating_foreign_pointer_throws) {
   int x = 0;
   EXPECT_THROW(alloc.deallocate(&x), std::runtime_error);
}

TEST_F(FUAllocatorTest, double_free_throws) {
   void *ptr = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   alloc.deallocate(ptr);

   EXPECT_THROW(alloc.deallocate(ptr), std::runtime_error);
}
