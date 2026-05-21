#ifndef FUSION_TESTS_AUTODIFF_TEST_BUILDERS_H
#define FUSION_TESTS_AUTODIFF_TEST_BUILDERS_H

#include <vector>

#include "Fusion/autodiff/ADTensor.hpp"
#include "Fusion/core/tensor/RawTensor.hpp"
#include "Fusion/device/Device.h"

namespace test_builders {

inline Device cpu_device() {
   return Device{DeviceType::CPU, 0};
}

inline std::vector<std::size_t> shape_2x3() {
   return {2, 3};
}

inline std::vector<std::size_t> shape_2() {
   return {2};
}

inline DType float32_dtype() {
   return DType::FLOAT32;
}

inline std::vector<float> data_square_inputs() {
   return {1.0, 4.0, 9.0, 16.0, 25.0, 36.0};
}

inline std::vector<float> data_linear_2x3() {
   return {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
}

inline std::vector<float> data_ones_2x3() {
   return {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
}

inline std::vector<float> data_sqrt_expected() {
   return {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
}

inline std::vector<float> data_square_plus_ones_expected() {
   return {2.0, 5.0, 10.0, 17.0, 26.0, 37.0};
}

inline std::vector<float> data_linear_times_two_expected() {
   return {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
}

inline std::vector<float> data_sum_axis1_expected() {
   return {6.0, 15.0};
}


inline RawTensor<float> raw_2x3(const std::vector<float>& data) {
   return RawTensor<float>(shape_2x3(), data, float32_dtype(), cpu_device());
}

inline RawTensor<float> raw_2(const std::vector<float>& data) {
   return RawTensor<float>(shape_2(), data, float32_dtype(), cpu_device());
}

inline ADTensor<float> ad_2x3(const std::vector<float>& data,
                              bool requires_grad = false) {
   return ADTensor<float>(shape_2x3(), data, float32_dtype(), cpu_device(),
                          requires_grad);
}

inline ADTensor<float> ad_2(const std::vector<float>& data,
                            bool requires_grad = false) {
   return ADTensor<float>(shape_2(), data, float32_dtype(), cpu_device(),
                          requires_grad);
}


inline ADTensor<float> ad_square_inputs(bool requires_grad = false) {
   return ad_2x3(data_square_inputs(), requires_grad);
}

inline ADTensor<float> ad_linear_inputs(bool requires_grad = false) {
   return ad_2x3(data_linear_2x3(), requires_grad);
}

inline ADTensor<float> ad_ones_inputs(bool requires_grad = false) {
   return ad_2x3(data_ones_2x3(), requires_grad);
}


inline RawTensor<float> raw_sqrt_expected() {
   return raw_2x3(data_sqrt_expected());
}

inline RawTensor<float> raw_square_plus_ones_expected() {
   return raw_2x3(data_square_plus_ones_expected());
}

inline RawTensor<float> raw_linear_times_two_expected() {
   return raw_2x3(data_linear_times_two_expected());
}

inline RawTensor<float> raw_sum_axis1_expected() {
   return raw_2(data_sum_axis1_expected());
}

} // namespace test_builders

#endif // FUSION_TESTS_AUTODIFF_TEST_BUILDERS_H