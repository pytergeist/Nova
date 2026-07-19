#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "Fusion/core/tensor/DenseTensor.hpp"

TEST(ExecutionCPUContractionTest,
     tag_fallback_contraction_accumulates_products) {
   const float a[] = {1., 90., 2., 33., 3., 99.};
   const float b[] = {4., 67., 5., 14., 7., 88.};
   float out[] = {1.};

   fusion::execution::cpu::detail::contraction_scalar_fallback<float,
                                                               MultiplySIMD>(
       out, a, b,
       0, // accumulate into single output element
       2, // a stride
       2, // b stride
       3  // len
   );
   EXPECT_EQ(out[0], 1. + 1. * 4. + 2. * 5. + 3. * 7.);
}

TEST(ExecutionCPUContractionTest,
     contraction_tag_computes_matrix_multiplication_result) {
   DenseTensor<float> a({2, 4},
                        std::vector<float>{
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                        },
                        DType::FLOAT32, Device{DeviceType::CPU, 0});

   DenseTensor<float> b({4, 3},
                        std::vector<float>{
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                        },
                        DType::FLOAT32, Device{DeviceType::CPU, 0});

   DenseTensor<float> out({2, 3}, DType::FLOAT32, Device{DeviceType::CPU, 0});

   fusion::fuir::OperandLabelBinding binding{
       .op_axis_labels =
           {
               {0, 1}, // Out: [i, j]
               {0, 2}, // A:     [i, k]
               {2, 1}, // B:     [k, j]
           },
       .out_labels = {0, 1},
   };
   fusion::planning::ContractionContext meta =
       fusion::planning::make_contraction_context_einsum(a, b, binding);

   fusion::execution::cpu::contraction<float, MultiplySIMD, MultiplySIMD>(
       a, b, meta, out);

   EXPECT_FLOAT_EQ(out[0], 70.);
   EXPECT_FLOAT_EQ(out[1], 80.);
   EXPECT_FLOAT_EQ(out[2], 90.);
   EXPECT_FLOAT_EQ(out[3], 158.);
   EXPECT_FLOAT_EQ(out[4], 184.);
   EXPECT_FLOAT_EQ(out[5], 210.);
}