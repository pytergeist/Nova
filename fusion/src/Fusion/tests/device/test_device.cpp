#include <gtest/gtest.h>

#include "Fusion/device/Device.h"

TEST(DeviceTest, cpu_device_reports_cpu_type_and_index) {
   Device d(DeviceType::CPU, 0);

   EXPECT_EQ(d.type(), DeviceType::CPU);
   EXPECT_EQ(d.idx(), 0);

   EXPECT_TRUE(d.is_cpu());
   EXPECT_FALSE(d.is_gpu());
   EXPECT_FALSE(d.is_cuda());
   EXPECT_FALSE(d.is_meta());
}

TEST(DeviceTest, cuda_device_reports_cuda_type_and_index) {
   Device d(DeviceType::CUDA, 0);

   EXPECT_EQ(d.type(), DeviceType::CUDA);
   EXPECT_EQ(d.idx(), 0);

   EXPECT_FALSE(d.is_cpu());
   EXPECT_TRUE(d.is_gpu());
   EXPECT_TRUE(d.is_cuda());
   EXPECT_FALSE(d.is_meta());
}

TEST(DeviceTest, metal_device_reports_metal_type_and_index) {
   Device d(DeviceType::METAL, 0);

   EXPECT_EQ(d.type(), DeviceType::METAL);
   EXPECT_EQ(d.idx(), 0);

   EXPECT_FALSE(d.is_cpu());
   EXPECT_TRUE(d.is_gpu());
   EXPECT_FALSE(d.is_cuda());
   EXPECT_TRUE(d.is_meta());
}

TEST(DeviceTest, equality_compares_type_and_index) {
   Device a(DeviceType::CUDA, 0);
   Device b(DeviceType::CUDA, 0);
   Device c(DeviceType::CUDA, 1);
   Device d(DeviceType::CPU, 0);

   EXPECT_TRUE(a == b);
   EXPECT_FALSE(a == c);
   EXPECT_FALSE(a == d);
}

TEST(DeviceTest, cpu_device_with_non_zero_index_throws) {
   EXPECT_THROW((Device(DeviceType::CPU, 1)), std::runtime_error);
}

TEST(DeviceTest, cpu_device_with_negative_index_throws) {
   EXPECT_THROW((Device(DeviceType::CPU, -1)), std::runtime_error);
}

TEST(DeviceTest, cuda_device_with_zero_index_is_valid) {
   EXPECT_NO_THROW((Device(DeviceType::CUDA, 0)));
}

TEST(DeviceTest, cuda_device_with_positive_index_is_valid) {
   EXPECT_NO_THROW((Device(DeviceType::CUDA, 1)));
}

TEST(DeviceTest, cuda_device_with_negative_index_throws) {
   EXPECT_THROW((Device(DeviceType::CUDA, -1)), std::runtime_error);
}

TEST(DeviceTest, metal_device_with_zero_index_is_valid) {
   EXPECT_NO_THROW((Device(DeviceType::METAL, 0)));
}

TEST(DeviceTest, metal_device_with_positive_index_is_valid) {
   EXPECT_NO_THROW((Device(DeviceType::METAL, 1)));
}

TEST(DeviceTest, metal_device_with_negative_index_throws) {
   EXPECT_THROW((Device(DeviceType::METAL, -1)), std::runtime_error);
}