#include <gtest/gtest.h>

#include "Fusion/storage/TensorBuffer.hpp"
#include "counting_allocator.h"

TEST(TensorBufferTest, DefaultConstructedBufferIsEmpty) {
   TensorBuffer buffer;

   EXPECT_TRUE(buffer.empty());
   EXPECT_EQ(buffer.size_bytes(), 0);
   EXPECT_FALSE(static_cast<bool>(buffer));
   EXPECT_EQ(buffer.data(), nullptr);
}

TEST(TensorBufferTest, AllocateWithCreatesNonEmptyBuffer) {
   CountingAllocator alloc;
   TensorBuffer buffer =
       TensorBuffer::allocate_with(&alloc, 128, Alignment{64});

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
   TensorBuffer buffer =
       TensorBuffer::allocate_elements_with<float>(&alloc, 16, Alignment{64});

   EXPECT_FALSE(buffer.empty());
   EXPECT_EQ(buffer.size_bytes(), 16 * sizeof(float));
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
   TensorBuffer buffer =
       TensorBuffer::allocate_elements_with<float>(&alloc, 4, Alignment{64});
   std::vector<int> src{1, 2, 3, 4};
   buffer.copy_from(src);

   const int *ptr = buffer.data<int>();
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

   const int *ptr = buffer.data<int>();
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

TEST(TensorBufferTest, DataPtrRespectsElementOffset) {
   CountingAllocator alloc;
   TensorBuffer buffer =
       TensorBuffer::allocate_elements_with<int>(&alloc, 4, Alignment{64});
   std::vector<int> src{1, 2, 3, 4};
   buffer.copy_from(src);
   const int *base = buffer.data_ptr<int>(0);
   const int *offset = buffer.data_ptr<int>(2);

   ASSERT_NE(base, nullptr);
   ASSERT_NE(offset, nullptr);
   EXPECT_EQ(*base, 1);
   EXPECT_EQ(*offset, 3);
   EXPECT_EQ(offset, base + 2);
}

TEST(TensorBufferTest, BeginAndEndSpanWholeBuffer) {
   CountingAllocator alloc;
   TensorBuffer buffer =
       TensorBuffer::allocate_elements_with<int>(&alloc, 3, Alignment{64});
   std::vector<int> src{1, 2, 3};
   buffer.copy_from(src);

   const int *begin = buffer.begin<int>();
   const int *end = buffer.end<int>();
   ASSERT_NE(begin, nullptr);
   ASSERT_NE(end, nullptr);
   EXPECT_EQ(begin[0], 1);
   EXPECT_EQ(begin[1], 2);
   EXPECT_EQ(begin[2], 3);
   EXPECT_EQ(end - begin, 3);
}

TEST(TensorBufferTest, DestructorReturnsMemoryToAllocator) {
   CountingAllocator alloc;
   void *raw_ptr = nullptr;

   {
      TensorBuffer buffer =
          TensorBuffer::allocate_with(&alloc, 128, Alignment{64});
      raw_ptr = buffer.data();
      EXPECT_EQ(alloc.allocate_calls(), 1);
      EXPECT_EQ(alloc.deallocate_calls(), 0);
      EXPECT_EQ(alloc.active_allocations(), 1);
   }

   EXPECT_EQ(alloc.deallocate_calls(), 1);
   EXPECT_EQ(alloc.active_allocations(), 0);
   EXPECT_EQ(alloc.last_deallocated_ptr(), raw_ptr);
}

TEST(TensorBufferTest, MoveConstructionTransfersOwnershipWithoutDoubleFree) {
   CountingAllocator alloc;
   void *raw_ptr = nullptr;
   {
      TensorBuffer buffer =
          TensorBuffer::allocate_with(&alloc, 128, Alignment{64});
      raw_ptr = buffer.data();
      TensorBuffer moved(std::move(buffer));
      EXPECT_EQ(alloc.allocate_calls(), 1);
      EXPECT_EQ(alloc.deallocate_calls(), 0);
      EXPECT_EQ(alloc.active_allocations(), 1);
   }
   EXPECT_EQ(alloc.deallocate_calls(), 1);
   EXPECT_EQ(alloc.active_allocations(), 0);
}

TEST(TensorBufferTest, MoveAssignmentTransfersOwnershipWithoutDoubleFree) {
   CountingAllocator alloc;
   void *raw_ptr = nullptr;
   {
      TensorBuffer buffer =
          TensorBuffer::allocate_with(&alloc, 128, Alignment{64});
      raw_ptr = buffer.data();
      TensorBuffer moved;
      moved = std::move(buffer);
      EXPECT_EQ(alloc.allocate_calls(), 1);
      EXPECT_EQ(alloc.deallocate_calls(), 0);
      EXPECT_EQ(alloc.active_allocations(), 1);
   }
   EXPECT_EQ(alloc.deallocate_calls(), 1);
   EXPECT_EQ(alloc.active_allocations(), 0);
}

TEST(TensorBufferTest, SwapExchangesContentsAndSizes) {
   CountingAllocator alloc;

   TensorBuffer a =
       TensorBuffer::allocate_elements_with<int>(&alloc, 2, Alignment{64});
   TensorBuffer b =
       TensorBuffer::allocate_elements_with<int>(&alloc, 4, Alignment{64});

   std::vector<int> va{1, 2};
   std::vector<int> vb{1, 2, 3, 4};

   a.copy_from(va);
   b.copy_from(vb);

   void *a_ptr_before = a.data();
   void *b_ptr_before = b.data();

   a.swap(b);

   EXPECT_EQ(a.data(), b_ptr_before);
   EXPECT_EQ(b.data(), a_ptr_before);
   EXPECT_EQ(a.size<int>(), 4);
   EXPECT_EQ(b.size<int>(), 2);

   const int *a_data = a.data<int>();
   const int *b_data = b.data<int>();
   ASSERT_NE(a_data, nullptr);
   ASSERT_NE(b_data, nullptr);

   EXPECT_EQ(a_data[0], 1);
   EXPECT_EQ(a_data[1], 2);
   EXPECT_EQ(a_data[2], 3);
   EXPECT_EQ(a_data[3], 4);

   EXPECT_EQ(b_data[0], 1);
   EXPECT_EQ(b_data[1], 2);
}

TEST(TensorBufferTest, MultipleBuffersEachReleaseExactlyOnce) {
   CountingAllocator alloc;
   {
      TensorBuffer a = TensorBuffer::allocate_with(&alloc, 128, Alignment{64});
      TensorBuffer b = TensorBuffer::allocate_with(&alloc, 64, Alignment{64});
      TensorBuffer c = TensorBuffer::allocate_with(&alloc, 256, Alignment{64});
      EXPECT_EQ(alloc.allocate_calls(), 3);
      EXPECT_EQ(alloc.deallocate_calls(), 0);
      EXPECT_EQ(alloc.active_allocations(), 3);
   }
   EXPECT_EQ(alloc.deallocate_calls(), 3);
   EXPECT_EQ(alloc.active_allocations(), 0);
}

TEST(TensorBufferTest, DefaultConstructedBufferHasNoOwnership) {
   TensorBuffer buffer;
   EXPECT_EQ(buffer.use_count(), 0);
}

TEST(TensorBufferTest, AllocatedBufferHasSingleOwnership) {
   CountingAllocator alloc;
   TensorBuffer buffer = TensorBuffer::allocate_with(&alloc, 64, Alignment{64});
   EXPECT_EQ(buffer.use_count(), 1);
}

TEST(TensorBufferTest, MoveConstructionTransfersSingleOwnership) {
   CountingAllocator alloc;

   TensorBuffer buffer =
       TensorBuffer::allocate_with(&alloc, 128, Alignment{64});
   EXPECT_EQ(buffer.use_count(), 1);
   TensorBuffer moved(std::move(buffer));
   EXPECT_EQ(moved.use_count(), 1);
   EXPECT_EQ(buffer.use_count(), 0);
}

TEST(TensorBufferTest, MoveAssignmentTransfersSingleOwnership) {
   CountingAllocator alloc;

   TensorBuffer buffer =
       TensorBuffer::allocate_with(&alloc, 128, Alignment{64});
   EXPECT_EQ(buffer.use_count(), 1);
   TensorBuffer moved;
   moved = std::move(buffer);
   EXPECT_EQ(moved.use_count(), 1);
   EXPECT_EQ(buffer.use_count(), 0);
}
