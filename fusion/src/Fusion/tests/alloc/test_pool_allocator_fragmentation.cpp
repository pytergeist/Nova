#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "Fusion/alloc/AllocTypes.h"
#include "Fusion/alloc/FUAllocator.h"
#include "Fusion/alloc/Pool.h"

class FUAllocatorFragmentationTest : public ::testing::Test {
 protected:
   FUAllocator alloc;
};

TEST_F(FUAllocatorFragmentationTest,
       alternating_free_pattern_leaves_fragmented_free_space) {
   std::vector<void *> ptrs;
   for (int i = 0; i < 8; ++i) {
      ptrs.push_back(alloc.allocate(64, Alignment{64}));
      ASSERT_NE(ptrs.back(), nullptr);
   }

   for (int i = 0; i < 8; i += 2) {
      alloc.deallocate(ptrs[i]);
   }

   int free_64_chunks = 0;
   for (const auto &chunk : alloc.chunks()) {
      if (chunk.size == 64 && !chunk.in_use) {
         free_64_chunks++;
      }
   }

   EXPECT_GE(free_64_chunks, 4);
}

TEST_F(FUAllocatorFragmentationTest,
       free_order_should_not_affect_coalescing_outcome) {
   void *seed = alloc.allocate(128, Alignment{64});
   ASSERT_NE(seed, nullptr);

   void *ptr1 = alloc.allocate(64, Alignment{64});
   void *ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr1, nullptr);
   ASSERT_NE(ptr2, nullptr);

   alloc.deallocate(ptr2);
   alloc.deallocate(ptr1);

   int free_64 = 0;
   int free_128 = 0;

   for (const auto &chunk : alloc.chunks()) {
      if (chunk.size == 0 || chunk.in_use) {
         continue;
      }
      if (chunk.size == 64) {
         free_64++;
      }
      if (chunk.size == 128) {
         free_128++;
      }
   }

   EXPECT_EQ(free_64, 0);
   EXPECT_EQ(free_128, 1);
}

TEST_F(FUAllocatorFragmentationTest,
       large_allocation_reuese_coalesced_space_without_growing_pool) {
   void *seed = alloc.allocate(128, Alignment{64});
   ASSERT_NE(seed, nullptr);

   void *a = alloc.allocate(64, Alignment{64});
   void *b = alloc.allocate(64, Alignment{64});
   ASSERT_NE(a, nullptr);
   ASSERT_NE(b, nullptr);

   alloc.deallocate(a);
   alloc.deallocate(b);

   const std::size_t chunk_count_before = alloc.chunks().size();

   void *big = alloc.allocate(128, Alignment{64});
   ASSERT_NE(big, nullptr);

   const std::size_t chunk_count_after = alloc.chunks().size();

   EXPECT_EQ(chunk_count_after, chunk_count_before);
}