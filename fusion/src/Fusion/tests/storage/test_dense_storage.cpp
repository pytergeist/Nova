#include <gtest/gtest.h>
#include <vector>

#include "counting_allocator.h"

#include "Fusion/core/device/Device.h"
#include "Fusion/core/memory/storage/DenseStorage.hpp"

TEST(DenseStorageTest, construct_from_count_allocates_requested_elements) {
   CountingAllocator alloc;
   std::vector<std::size_t> shape{2, 3};
   NDTensorStorage<float> storage(shape, 6, Device{DeviceType::CPU, 0}, &alloc);

   EXPECT_NE(storage.data_ptr(), nullptr);
   EXPECT_EQ(storage.size(), 6);
   EXPECT_EQ(storage.data().size_bytes(), 6U * sizeof(float));

   EXPECT_EQ(alloc.allocate_calls(), 1);
   EXPECT_EQ(alloc.last_size(), 6 * sizeof(float));
}

TEST(DenseStorageTest, size_returns_element_count) {
   CountingAllocator alloc;
   std::vector<std::size_t> shape{4};
   std::vector<float> values{2, 3, 5, 7};
   NDTensorStorage<float> storage(shape, values, Device{DeviceType::CPU, 0},
                                  &alloc);
   EXPECT_EQ(storage.size(), values.size());
}

TEST(DenseStorageTest, device_returns_constructor_device) {
   CountingAllocator alloc;
   std::vector<std::size_t> shape{4};
   std::vector<float> values{2, 3, 5, 7};
   Device device{DeviceType::CPU, 0};
   NDTensorStorage<float> storage(shape, values, device, &alloc);
   EXPECT_EQ(storage.device(), device);
}

TEST(DenseStorageTest, data_accessor_returns_underlying_buffer) {
   CountingAllocator alloc;
   std::vector<std::size_t> shape{4};
   std::vector<float> values{2, 3, 5, 7};
   Device device{DeviceType::CPU, 0};
   NDTensorStorage<float> storage(shape, values, device, &alloc);

   TensorBuffer &buffer = storage.data();

   EXPECT_FALSE(buffer.empty());
   EXPECT_EQ(buffer.size<float>(), values.size());
   EXPECT_EQ(buffer.data<float>(), storage.data_ptr());
}

TEST(DenseStorageTest, const_data_ptr_returns_typed_pointer) {
   CountingAllocator alloc;
   std::vector<std::size_t> shape{4};
   std::vector<float> values{2, 3, 5, 7};
   Device device{DeviceType::CPU, 0};
   NDTensorStorage<float> storage(shape, values, device, &alloc);

   const float *ptr = storage.data_ptr();
   EXPECT_NE(ptr, nullptr);
   EXPECT_EQ(ptr[0], values[0]);
   EXPECT_EQ(ptr[1], values[1]);
   EXPECT_EQ(ptr[2], values[2]);
   EXPECT_EQ(ptr[3], values[3]);
}

TEST(DenseStorageTest, destructor_returns_memory_to_allocator) {
   CountingAllocator alloc;
   {
      std::vector<std::size_t> shape{4};
      std::vector<float> values{2, 3, 5, 7};
      Device device{DeviceType::CPU, 0};
      NDTensorStorage<float> storage(shape, values, device, &alloc);

      EXPECT_EQ(alloc.allocate_calls(), 1);
      EXPECT_EQ(alloc.deallocate_calls(), 0);
      EXPECT_EQ(alloc.active_allocations(), 1);
   }

   EXPECT_EQ(alloc.deallocate_calls(), 1);
   EXPECT_EQ(alloc.active_allocations(), 0);
}
