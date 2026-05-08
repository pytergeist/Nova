#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "Fusion/alloc/AllocTypes.h"
#include "Fusion/alloc/FUAllocator.h"
#include "Fusion/alloc/Pool.h"

class FUAllocatorBucketTest : public ::testing::Test {
 protected:
   FUAllocator alloc;
};

static bool contains_id(const std::vector<ChunkID> &ids, ChunkID id) {
   return std::find(ids.begin(), ids.end(), id) != ids.end();
}

TEST_F(FUAllocatorBucketTest, freed_chunk_is_inserted_into_matching_bucket) {
   void *ptr = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr, nullptr);
   ChunkID id = kInvalidChunkID;
   for (Chunk &chunk : alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         id = chunk.chunk_id;
         break;
      }
   }
   ASSERT_NE(id, kInvalidChunkID);
   alloc.deallocate(ptr);

   std::vector<ChunkID> free_chunks = alloc.get_free_chunks(64);
   EXPECT_TRUE(contains_id(free_chunks, id));
}

TEST_F(FUAllocatorBucketTest, reallocated_chunk_is_removed_from_free_bucket) {
   void *ptr = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr, nullptr);
   ChunkID id = kInvalidChunkID;
   for (Chunk &chunk : alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         id = chunk.chunk_id;
         break;
      }
   }
   ASSERT_NE(id, kInvalidChunkID);
   alloc.deallocate(ptr);

   std::vector<ChunkID> free_before = alloc.get_free_chunks(64);

   void *ptr2 = alloc.allocate(64, Alignment{64});
   ASSERT_NE(ptr2, nullptr);

   std::vector<ChunkID> free_after = alloc.get_free_chunks(64);

   EXPECT_FALSE(contains_id(free_after, id));
}

TEST_F(FUAllocatorBucketTest, missing_bucket_returns_empty_vector) {
   std::vector<ChunkID> free_chunks = alloc.get_free_chunks(64);
   EXPECT_TRUE(free_chunks.empty());
}

TEST_F(FUAllocatorBucketTest,
       larger_free_bucket_can_satisfy_smaller_request) {
   void *ptr = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   alloc.deallocate(ptr);

   std::vector<ChunkID> free_128_chunks = alloc.get_free_chunks(128);
   ASSERT_FALSE(free_128_chunks.empty());

   void *sptr = alloc.allocate(64, Alignment{64});
   ASSERT_NE(sptr, nullptr);
   EXPECT_EQ(sptr, ptr);
}

TEST_F(FUAllocatorBucketTest, larger_free_bucket_split_by_smaller_request) {
   void *ptr = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   alloc.deallocate(ptr);

   std::vector<ChunkID> free_128_chunks_before = alloc.get_free_chunks(128);
   ASSERT_FALSE(free_128_chunks_before.empty());

   void *sptr = alloc.allocate(64, Alignment{64});
   ASSERT_NE(sptr, nullptr);
   std::vector<ChunkID> free_128_chunks_after = alloc.get_free_chunks(128);
   std::vector<ChunkID> free_64_chunks = alloc.get_free_chunks(64);
   EXPECT_TRUE(free_128_chunks_after.empty());
   EXPECT_EQ(free_64_chunks.size(), 1);
}

TEST_F(FUAllocatorBucketTest, exact_size_chunk_returns_to_expected_bucket) {
   void *ptr = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr, nullptr);

   ChunkID id = kInvalidChunkID;
   for (Chunk &chunk : alloc.chunks()) {
      if (chunk.ptr == ptr && chunk.size > 0) {
         id = chunk.chunk_id;
         break;
      }
   }
   ASSERT_NE(id, kInvalidChunkID);
   alloc.deallocate(ptr);
   std::vector<ChunkID> free_chunks = alloc.get_free_chunks(128);
   EXPECT_TRUE(contains_id(free_chunks, id));
}

TEST_F(FUAllocatorBucketTest, bucket_contains_only_free_chunks) {
   void *ptr1 = alloc.allocate(64, Alignment{64});
   void *ptr2 = alloc.allocate(128, Alignment{64});
   ASSERT_NE(ptr1, nullptr);
   ASSERT_NE(ptr2, nullptr);

   std::vector<std::size_t> buckets{64, 128};

   for (std::size_t bucket_size : buckets) {
      std::vector<ChunkID> free_chunks = alloc.get_free_chunks(bucket_size);
      for (ChunkID id : free_chunks) {
         Chunk chunk = alloc.chunks().at(id);
         EXPECT_FALSE(chunk.in_use);
         EXPECT_EQ(chunk.size, bucket_size);
      }
   }
}

TEST_F(FUAllocatorBucketTest,
       freed_bucket_matches_actual_free_chunks_of_bucket_size) {
   void *p1 = alloc.allocate(64, Alignment{64});
   void *p2 = alloc.allocate(128, Alignment{64});
   ASSERT_NE(p1, nullptr);
   ASSERT_NE(p2, nullptr);

   alloc.deallocate(p1);
   alloc.deallocate(p2);

   std::vector<std::size_t> buckets{64, 128};

   for (std::size_t bucket_size : buckets) {
      std::vector<ChunkID> expected;
      for (const auto &chunk : alloc.chunks()) {
         if (chunk.size == 0 || chunk.in_use) {
            continue;
         }
         if ((std::size_t{1} << (std::bit_width(chunk.size) - 1)) ==
             bucket_size) {
            expected.push_back(chunk.chunk_id);
         }
      }

      std::vector<ChunkID> actual = alloc.get_free_chunks(bucket_size);

      std::sort(expected.begin(), expected.end());
      std::sort(actual.begin(), actual.end());

      EXPECT_EQ(actual, expected);
   }
}