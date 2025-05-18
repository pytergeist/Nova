 // tensor_ops_test.cpp

 #include "../tensor/tensor.h"
 #include <cmath>
 #include <gtest/gtest.h>
 #include <stdexcept>
 #include <vector>

 //------------------------------------------------------------------------------
 // Helper: Create a Tensor<T> from vector data and a shape.
 // If the shape has one element, we treat it as an n×1 tensor.
 // If the shape has two elements, we create an n×m matrix.
 // Throws std::invalid_argument if the data size does not match.
 template <typename T>
 Tensor<T> create_tensor(const std::vector<T> &data,
                         const std::vector<size_t> &shape) {


     return Tensor<T>(shape, data, Device::CPU);
 };
// //------------------------------------------------------------------------------
// // Unary Operations Tests
//
// TEST(TensorOpsTest, UnarySqrtTest) {
//   std::vector<double> data = {4.0, 9.0, 16.0};
//   std::vector<size_t> shape = {3}; // 3×1 tensor.
//   Tensor<double> tensor = create_tensor(data, shape);
//   Tensor<double> result = tensor.sqrt();
//
//   std::vector<double> expected = {2.0, 3.0, 4.0};
//   std::vector<double> resultVec = flatten(result);
//   ASSERT_EQ(resultVec.size(), expected.size());
//   for (size_t i = 0; i < expected.size(); ++i) {
//     EXPECT_NEAR(resultVec[i], expected[i], 1e-6);
//   }
// }
//
// TEST(TensorOpsTest, UnaryExpTest) {
//   std::vector<double> data = {0.0, 1.0};
//   std::vector<size_t> shape = {2}; // 2×1 tensor.
//   Tensor<double> tensor = create_tensor(data, shape);
//   Tensor<double> result = tensor.exp();
//
//   std::vector<double> expected = {std::exp(0.0), std::exp(1.0)};
//   std::vector<double> resultVec = flatten(result);
//   ASSERT_EQ(resultVec.size(), expected.size());
//   for (size_t i = 0; i < expected.size(); ++i) {
//     EXPECT_NEAR(resultVec[i], expected[i], 1e-6);
//   }
// }
//
// TEST(TensorOpsTest, UnaryLogTest) {
//   std::vector<double> data = {std::exp(1.0), std::exp(2.0)};
//   std::vector<size_t> shape = {2}; // 2×1 tensor.
//   Tensor<double> tensor = create_tensor(data, shape);
//   Tensor<double> result = tensor.log();
//
//   std::vector<double> expected = {1.0, 2.0};
//   std::vector<double> resultVec = flatten(result);
//   ASSERT_EQ(resultVec.size(), expected.size());
//   for (size_t i = 0; i < expected.size(); ++i) {
//     EXPECT_NEAR(resultVec[i], expected[i], 1e-6);
//   }
// }
//
 //------------------------------------------------------------------------------
 // Binary Operations Tests

 TEST(TensorOpsTest, BinaryOperationsTest) {
   std::vector<double> data1 = {1.0, 2.0, 3.0};
   std::vector<double> data2 = {4.0, 5.0, 6.0};
   std::vector<size_t> shape = {3}; // 3×1 tensors.
   Tensor<double> t1 = create_tensor(data1, shape);
   Tensor<double> t2 = create_tensor(data2, shape);

   // Addition
   Tensor<double> addResult = t1 + t2;
   std::vector<double> expectedAdd = {5.0, 7.0, 9.0};
   std::vector<double> addVec = flatten(addResult);
   for (size_t i = 0; i < expectedAdd.size(); ++i) {
     EXPECT_NEAR(addVec[i], expectedAdd[i], 1e-6);
   }

   // Subtraction
   Tensor<double> subResult = t2 - t1;
   std::vector<double> expectedSub = {3.0, 3.0, 3.0};
   std::vector<double> subVec = flatten(subResult);
   for (size_t i = 0; i < expectedSub.size(); ++i) {
     EXPECT_NEAR(subVec[i], expectedSub[i], 1e-6);
   }

   // Elementwise Multiplication
   Tensor<double> mulResult = t1 * t2;
   std::vector<double> expectedMul = {4.0, 10.0, 18.0};
   std::vector<double> mulVec = flatten(mulResult);
   for (size_t i = 0; i < expectedMul.size(); ++i) {
     EXPECT_NEAR(mulVec[i], expectedMul[i], 1e-6);
   }

   // Elementwise Division
   Tensor<double> divResult = t2 / t1;
   std::vector<double> expectedDiv = {4.0, 2.5, 2.0};
   std::vector<double> divVec = flatten(divResult);
   for (size_t i = 0; i < expectedDiv.size(); ++i) {
     EXPECT_NEAR(divVec[i], expectedDiv[i], 1e-6);
   }

   // Pow: Note that Tensor::pow takes a scalar exponent.
   // Here, we raise each element of t1 to the 4th power.
   Tensor<double> powResult = t1.pow(4.0);
   std::vector<double> expectedPow;
   for (double d : data1) {
     expectedPow.push_back(std::pow(d, 4.0));
   }
   std::vector<double> powVec = flatten(powResult);
   for (size_t i = 0; i < expectedPow.size(); ++i) {
     EXPECT_NEAR(powVec[i], expectedPow[i], 1e-6);
   }
 }
//
// //------------------------------------------------------------------------------
// // Binary Operation With "Scalar"
// // Since our Tensor class does not support direct scalar–tensor operations,
// // we simulate a scalar by creating a tensor of matching shape with all elements
// // equal.
// TEST(TensorOpsTest, BinaryOpWithScalarTest) {
//   std::vector<double> data = {2.0, 4.0, 6.0};
//   std::vector<size_t> shape = {3}; // 3×1 tensor.
//   Tensor<double> tensor = create_tensor(data, shape);
//
//   // Create a tensor (of same shape) filled with 2.0.
//   std::vector<double> scalarData(shape[0], 2.0);
//   Tensor<double> scalarTensor = create_tensor(scalarData, shape);
//
//   // Test addition: scalar + tensor.
//   Tensor<double> addResult = scalarTensor + tensor;
//   std::vector<double> expectedAdd;
//   for (double d : data) {
//     expectedAdd.push_back(2.0 + d);
//   }
//   std::vector<double> addVec = flatten(addResult);
//   for (size_t i = 0; i < expectedAdd.size(); ++i) {
//     EXPECT_NEAR(addVec[i], expectedAdd[i], 1e-6);
//   }
//
//   // Test subtraction: tensor - scalar.
//   Tensor<double> subResult = tensor - scalarTensor;
//   std::vector<double> expectedSub;
//   for (double d : data) {
//     expectedSub.push_back(d - 2.0);
//   }
//   std::vector<double> subVec = flatten(subResult);
//   for (size_t i = 0; i < expectedSub.size(); ++i) {
//     EXPECT_NEAR(subVec[i], expectedSub[i], 1e-6);
//   }
// }
//
// //------------------------------------------------------------------------------
// // Mismatched Sizes Test
//
// TEST(TensorOpsTest, MismatchedSizeTest) {
//   std::vector<double> data1 = {1.0, 2.0};
//   std::vector<double> data2 = {3.0, 4.0, 5.0};
//   std::vector<size_t> shape1 = {2}; // 2×1 tensor.
//   std::vector<size_t> shape2 = {3}; // 3×1 tensor.
//   Tensor<double> t1 = create_tensor(data1, shape1);
//   Tensor<double> t2 = create_tensor(data2, shape2);
//
//   EXPECT_THROW(
//       {
//         auto result = t1 + t2;
//         (void)result;
//       },
//       std::invalid_argument);
// }
//
// //------------------------------------------------------------------------------
// // Matrix Multiplication Tests
//
// TEST(TensorOpsTest, MatrixMultiplication2DTest) {
//   // Define two 2x2 matrices:
//   // A = [1, 2;
//   //      3, 4]
//   // B = [5, 6;
//   //      7, 8]
//   // Expected: A.matmul(B) = [19, 22;
//   //                           43, 50]
//   std::vector<double> dataA = {1, 2, 3, 4};
//   std::vector<size_t> shapeA = {2, 2};
//   Tensor<double> A = create_tensor(dataA, shapeA);
//
//   std::vector<double> dataB = {5, 6, 7, 8};
//   std::vector<size_t> shapeB = {2, 2};
//   Tensor<double> B = create_tensor(dataB, shapeB);
//
//   Tensor<double> result = A.matmul(B);
//   std::vector<double> expected = {19, 22, 43, 50};
//   std::vector<double> resVec = flatten(result);
//   ASSERT_EQ(resVec.size(), expected.size());
//   for (size_t i = 0; i < expected.size(); ++i) {
//     EXPECT_NEAR(resVec[i], expected[i], 1e-6);
//   }
// }
//
// TEST(TensorOpsTest, MatrixMultiplication1DTest) {
//   // For a 2x3 matrix A and a 3×1 tensor v:
//   // A = [1, 2, 3;
//   //      4, 5, 6]
//   // v = [1; 1; 1]  (represented as a 3×1 tensor)
//   // Expected: A.matmul(v) = [6; 15]
//   std::vector<double> dataA = {1, 2, 3, 4, 5, 6};
//   std::vector<size_t> shapeA = {2, 3};
//   Tensor<double> A = create_tensor(dataA, shapeA);
//
//   std::vector<double> dataV = {1, 1, 1};
//   // Using a 1D shape for v creates a 3×1 tensor.
//   std::vector<size_t> shapeV = {3};
//   Tensor<double> v = create_tensor(dataV, shapeV);
//
//   Tensor<double> result = A.matmul(v);
//   std::vector<double> expected = {6, 15};
//   std::vector<double> resVec = flatten(result);
//   ASSERT_EQ(resVec.size(), expected.size());
//   for (size_t i = 0; i < expected.size(); ++i) {
//     EXPECT_NEAR(resVec[i], expected[i], 1e-6);
//   }
// }
//
// TEST(TensorOpsTest, MatrixMultiplicationInvalidShapeTest) {
//   std::vector<double> data = {1, 2, 3};
//   std::vector<size_t> shape = {3}; // 3×1 tensor.
//   Tensor<double> tensor = create_tensor(data, shape);
//
//   EXPECT_THROW(
//       {
//         auto result = tensor.matmul(tensor);
//         (void)result;
//       },
//       std::invalid_argument);
// }
//
// //------------------------------------------------------------------------------
// // main() for Google Test
//
// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
