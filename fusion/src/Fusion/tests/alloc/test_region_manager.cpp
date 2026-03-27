#include <gtest/gtest.h>
#include <vector>
#include <cstddef>

#include "Fusion/alloc/AllocTypes.h"
#include "Fusion/alloc/Pool.h"
#include "Fusion/alloc/BFCPoolAllocator.h"


TEST(RegionManagerTest, StartsWithNoRegions) {
   RegionManager rm;
   EXPECT_TRUE(rm.regions().empty());
}

TEST(RegionManagerTest, AddAllocatedRegionStoresRegionMetadata) {
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


TEST(RegionManagerTest, AddAllocatedRegionAssignsIncreasingRegionIds) {
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


TEST(RegionManagerTest, FindRegionForPtrReturnsRegionWhenPointerIsAtBase) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);
   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   Region& region = rm.find_region_for_ptr(buffer.data());

   EXPECT_EQ(region.ptr, buffer.data());
}

TEST(RegionManagerTest, FindRegionForPtrReturnsRegionWhenPointerIsInsideRegion) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);
   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   void* inner_ptr = buffer.data() + 32;

   Region& region = rm.find_region_for_ptr(inner_ptr);

   EXPECT_EQ(region.ptr, buffer.data());
}


TEST(RegionManagerTest, FindRegionForPtrThrowsWhenPointerIsAtEnd) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);
   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   void* end_ptr = buffer.data() + buffer.size();

   EXPECT_THROW(rm.find_region_for_ptr(end_ptr), std::runtime_error);
}


TEST(RegionManagerTest, FindRegionForPtrThrowsWhenPointerIsOutsideRegion) {
   RegionManager rm;
   std::vector<std::byte> buffer(128);
   rm.add_allocated_region(buffer.data(), buffer.size(), Alignment{64});

   int other = 0;

   EXPECT_THROW(rm.find_region_for_ptr(&other), std::runtime_error);
}


TEST(RegionManagerTest, FindRegionForPtrFindsCorrectRegionAmongMultipleRegions) {
   RegionManager rm;
   std::vector<std::byte> buffer1(32);
   std::vector<std::byte> buffer2(128);
   rm.add_allocated_region(buffer1.data(), buffer1.size(), Alignment{64});
   rm.add_allocated_region(buffer2.data(), buffer2.size(), Alignment{64});

   void* ptr1 = buffer1.data() + 10;
   Region& region1 = rm.find_region_for_ptr(ptr1);

   EXPECT_EQ(region1.ptr, buffer1.data());
   EXPECT_EQ(region1.size, 32);
   EXPECT_EQ(region1.region_id, 0);
}

TEST(RegionManagerTest, SetChunkIdThenGetChunkIdReturnsStoredValue) {
   RegionManager rm;
   int x = 0;

   rm.set_chunkid(&x, ChunkID{7});

   EXPECT_EQ(rm.get_chunkid_from_ptr(&x), ChunkID{7});
}

TEST(RegionManagerTest, SetChunkIdOverwritesExistingMapping) {
   RegionManager rm;
   int x = 0;

   rm.set_chunkid(&x, ChunkID{7});
   rm.set_chunkid(&x, ChunkID{9});

   EXPECT_EQ(rm.get_chunkid_from_ptr(&x), ChunkID{9});
}


TEST(RegionManagerTest, GetChunkIdFromUnknownPointerThrows) {
   RegionManager rm;

   int x = 0;

   EXPECT_THROW(rm.get_chunkid_from_ptr(&x), std::runtime_error);
}

TEST(RegionManagerTest, EraseChunkReturnsTrueWhenPointerWasTracked) {
   RegionManager rm;

   int x = 0;
   rm.set_chunkid(&x, ChunkID{7});
   EXPECT_TRUE(rm.erase_chunk(&x));
}

TEST(RegionManagerTest, EraseChunkRemovesPointerMapping) {
   RegionManager rm;
   int x = 0;
   rm.set_chunkid(&x, ChunkID{7});
   rm.erase_chunk(&x);
   EXPECT_THROW(rm.get_chunkid_from_ptr(&x), std::runtime_error);
}


TEST(RegionManagerTest, EraseChunkReturnsFalseWhenPointerWasNotTracked) {
   RegionManager rm;
   int x = 0;
   EXPECT_FALSE(rm.erase_chunk(&x));
}


