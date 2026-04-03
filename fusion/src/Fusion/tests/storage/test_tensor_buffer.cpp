#include <gtest/gtest.h>

#include "counting_allocator.h"
#include "Fusion/storage/TensorBuffer.hpp"


TEST(TensorBufferTest, DefaultConstructedBufferIsEmpty) {
   TensorBuffer buffer;

   EXPECT_TRUE(buffer.empty());
   EXPECT_EQ(buffer.size_bytes(), 0);
   EXPECT_FALSE(static_cast<bool>(buffer));
   EXPECT_EQ(buffer.data(), nullptr);
}

TEST(TensorBufferTest, AllocateWithCreatesNonEmptyBuffer) {
   CountingAllocator alloc;
   TensorBuffer buffer = TensorBuffer::allocate_with(&alloc, 128, Alignment{64});

   EXPECT_FALSE(buffer.empty());
   EXPECT_EQ(buffer.size_bytes(), 128);
   EXPECT_TRUE(static_cast<bool>(buffer));
   EXPECT_NE(buffer.data(), nullptr);

   EXPECT_EQ(alloc.allocate_calls(), 1);
   EXPECT_EQ(alloc.deallocate_calls(), 0);
   EXPECT_EQ(alloc.last_size(), 128);
   EXPECT_EQ(alloc.last_alignment(), 64);
   }


TEST(TensorBufferTest, AllocateElementsWithUsesElementCount) {
   CountingAllocator alloc;
   TensorBuffer buffer = TensorBuffer::allocate_elements_with<float>(&alloc, 16, Alignment{64});

   EXPECT_FALSE(buffer.empty());
   EXPECT_EQ(buffer.size_bytes(), 16*sizeof(float));
   EXPECT_EQ(buffer.size<float>(), 16);
   EXPECT_NE(buffer.data(), nullptr);
}

TEST(TensorBufferTest, ZeroSizeAllocationReturnsEmptyBuffer) {
   CountingAllocator alloc;
   TensorBuffer buffer = TensorBuffer::allocate_with(&alloc, 0, Alignment{64});

   EXPECT_TRUE(buffer.empty());
   EXPECT_EQ(buffer.size_bytes(), 0);
   EXPECT_FALSE(static_cast<bool>(buffer));
   EXPECT_EQ(buffer.data(), nullptr);
}

TEST(TensorBufferTest, CopyFromCopiesValuesIntoBuffer) {
   CountingAllocator alloc;
   TensorBuffer buffer = TensorBuffer::allocate_elements_with<float>(&alloc, 4, Alignment{64});
   std::vector<int> src{1, 2, 3, 4};
   buffer.copy_from(src);

   const int* ptr = buffer.data<int>();
   ASSERT_NE(ptr, nullptr);
   EXPECT_EQ(ptr[0], 1);
   EXPECT_EQ(ptr[1], 2);
   EXPECT_EQ(ptr[2], 3);
   EXPECT_EQ(ptr[3], 4);
}

TEST(TensorBufferTest, CopyFromWithOffsetCopiesToCorrectLocation) {
   CountingAllocator alloc;
   TensorBuffer buffer =
       TensorBuffer::allocate_elements_with<int>(&alloc, 5, Alignment{64});

   std::vector<int> initial{10, 20, 30, 40, 50};
   buffer.copy_from(initial);

   std::vector<int> patch{111, 222};
   buffer.copy_from(patch, 2);

   const int* ptr = buffer.data<int>();
   ASSERT_NE(ptr, nullptr);
   EXPECT_EQ(ptr[0], 10);
   EXPECT_EQ(ptr[1], 20);
   EXPECT_EQ(ptr[2], 111);
   EXPECT_EQ(ptr[3], 222);
   EXPECT_EQ(ptr[4], 50);
}


TEST(TensorBufferTest, CopyFromThrowsWhenDestinationTooSmall) {
   CountingAllocator alloc;
   TensorBuffer buffer =
       TensorBuffer::allocate_elements_with<int>(&alloc, 4, Alignment{64});

   std::vector<int> initial{1, 2, 3, 4, 5};

   EXPECT_THROW(buffer.copy_from(initial), std::out_of_range);
}


TEST(TensorBufferTest, CopyFromThrowsWhenSourceIsEmpty) {
   CountingAllocator alloc;
   TensorBuffer buffer =
       TensorBuffer::allocate_elements_with<int>(&alloc, 2, Alignment{64});

   std::vector<int> src{};

   EXPECT_THROW(buffer.copy_from(src), std::out_of_range);
}
