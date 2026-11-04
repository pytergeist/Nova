#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

#include "Fusion/core/memory/alloc/AllocTypes.h"
#include "Fusion/core/memory/alloc/FUAllocator.h"
#include "Fusion/core/memory/alloc/Pool.h"

TEST(RegionManagerTest, starts_with_no_regions) {
   RegionManager rm;
   EXPECT_TRUE(rm.regions().empty());
}

TEST(RegionManagerTest, add_allocated_region_stores_region_meta_data) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);

   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   std::vector<Region> regions = rm.regions();
   ASSERT_EQ(regions.size(), 1);

   EXPECT_EQ(regions[0].ptr, buffer.data());
   EXPECT_EQ(regions[0].size, 128);
   EXPECT_EQ(regions[0].region_id, 0);
   EXPECT_EQ(regions[0].alignment, Alignment{64});
}

TEST(RegionManagerTest, add_allocated_region_assignes_increasing_region_ids) {
   RegionManager rm;
   std::vector<std::byte> buffer1(128);
   std::vector<std::byte> buffer2(128);
   std::vector<std::byte> buffer3(128);

   rm.add_allocated_region(buffer1.data(), buffer1.size(), Alignment{64});
   rm.add_allocated_region(buffer2.data(), buffer2.size(), Alignment{64});
   rm.add_allocated_region(buffer3.data(), buffer3.size(), Alignment{64});

   std::vector<Region> regions = rm.regions();

   ASSERT_EQ(regions.size(), 3);

   EXPECT_EQ(regions[0].region_id, 0);
   EXPECT_EQ(regions[1].region_id, 1);
   EXPECT_EQ(regions[2].region_id, 2);
}

TEST(RegionManagerTest,
     find_region_for_ptr_returns_region_when_pointer_is_at_base) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);
   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   Region &region = rm.find_region_for_ptr(buffer.data());

   EXPECT_EQ(region.ptr, buffer.data());
}

TEST(RegionManagerTest,
     find_region_for_ptr_returns_region_when_pointer_is_inside_region) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);
   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   void *inner_ptr = buffer.data() + 32;

   Region &region = rm.find_region_for_ptr(inner_ptr);

   EXPECT_EQ(region.ptr, buffer.data());
}

TEST(RegionManagerTest, find_region_for_ptr_throws_when_pointer_is_at_end) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);
   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   void *end_ptr = buffer.data() + buffer.size();

   EXPECT_THROW(rm.find_region_for_ptr(end_ptr), std::runtime_error);
}

TEST(RegionManagerTest,
     find_region_for_ptr_throws_when_pointer_is_outisde_region) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);
   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   int other = 0;

   EXPECT_THROW(rm.find_region_for_ptr(&other), std::runtime_error);
}

TEST(RegionManagerTest,
     find_region_for_ptr_finds_correct_region_among_multiple_regions) {
   RegionManager rm;
   std::vector<std::byte> buffer1(32);
   std::vector<std::byte> buffer2(128);
   rm.add_allocated_region(buffer1.data(), buffer1.size(), Alignment{64});
   rm.add_allocated_region(buffer2.data(), buffer2.size(), Alignment{64});

   void *ptr1 = buffer1.data() + 10;
   Region &region1 = rm.find_region_for_ptr(ptr1);

   EXPECT_EQ(region1.ptr, buffer1.data());
   EXPECT_EQ(region1.size, 32);
   EXPECT_EQ(region1.region_id, 0);
}

TEST(RegionManagerTest, set_chunk_id_then_get_chunk_id_returns_stored_value) {
   RegionManager rm;
   int x = 0;

   rm.set_chunkid(&x, ChunkID{7});

   EXPECT_EQ(rm.get_chunkid_from_ptr(&x), ChunkID{7});
}

TEST(RegionManagerTest, set_chunk_id_overwrites_existing_mapping) {
   RegionManager rm;
   int x = 0;

   rm.set_chunkid(&x, ChunkID{7});
   rm.set_chunkid(&x, ChunkID{9});

   EXPECT_EQ(rm.get_chunkid_from_ptr(&x), ChunkID{9});
}

TEST(RegionManagerTest, get_chunk_id_from_unknown_pointer_throws) {
   RegionManager rm;

   int x = 0;

   EXPECT_THROW(rm.get_chunkid_from_ptr(&x), std::runtime_error);
}

TEST(RegionManagerTest, erase_chunk_returns_true_when_pointer_was_tracked) {
   RegionManager rm;

   int x = 0;
   rm.set_chunkid(&x, ChunkID{7});
   EXPECT_TRUE(rm.erase_chunk(&x));
}

TEST(RegionManagerTest, erase_chunk_removes_pointer_mapping) {
   RegionManager rm;
   int x = 0;
   rm.set_chunkid(&x, ChunkID{7});
   rm.erase_chunk(&x);
   EXPECT_THROW(rm.get_chunkid_from_ptr(&x), std::runtime_error);
}

TEST(RegionManagerTest,
     erase_chunk_returns_false_when_pointer_was_not_tracked) {
   RegionManager rm;
   int x = 0;
   EXPECT_FALSE(rm.erase_chunk(&x));
}
