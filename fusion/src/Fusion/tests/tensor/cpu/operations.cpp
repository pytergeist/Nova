// tensor_ops_test.cpp

#include "../../../Tensor.h"
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

//------------------------------------------------------------------------------
// Helper: Create a Tensor<T> from vector data and a shape.
// If the shape has one element, we treat it as an n×1 tensor.
// If the shape has two elements, we create an n×m matrix.
// Throws std::invalid_argument if the data size does not match.
using T = float;

template <typename T>
Tensor<T> create_tensor(const std::vector<T> &data,
                        const std::vector<std::size_t> &shape) {

   return Tensor<T>(std::move(shape), std::move(data), Device::CPU);
};
//------------------------------------------------------------------------------
// Unary Operations Tests

TEST(TensorOpsTest, UnarySqrtTest) {
   std::vector<T> data = {4.0, 9.0, 16.0};
   std::vector<std::size_t> shape = {3}; // 3×1 tensor.
   Tensor<T> tensor = create_tensor(data, shape);
   Tensor<T> result = tensor.sqrt();
   Tensor<T> expected = create_tensor(std::vector<T>{2.0, 3.0, 4.0}, shape);

   ASSERT_EQ(result.size(), expected.size());
   for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(result[i], expected[i], 1e-6);
   }
}

TEST(TensorOpsTest, UnaryExpTest) {
   std::vector<T> data = {0.0, 1.0};
   std::vector<std::size_t> shape = {2}; // 2×1 tensor.
   Tensor<T> tensor = create_tensor(data, shape);
   Tensor<T> result = tensor.exp();
   Tensor<T> expected =
       create_tensor(std::vector<T>{std::exp(0.0f), std::exp(1.0f)}, shape);
   ASSERT_EQ(result.size(), expected.size());
   for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(result[i], expected[i], 1e-6);
   }
}

TEST(TensorOpsTest, UnaryLogTest) {
   std::vector<T> data = {std::exp(1.0f), std::exp(2.0f)};
   std::vector<size_t> shape = {2}; // 2×1 tensor.
   Tensor<T> tensor = create_tensor(data, shape);
   Tensor<T> result = tensor.log();

   Tensor<T> expected = create_tensor(std::vector<T>{1.0, 2.0}, shape);
   ASSERT_EQ(result.size(), expected.size());
   for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(result[i], expected[i], 1e-6);
   }
}

////------------------------------------------------------------------------------
// Binary Operations Tests

TEST(TensorOpsTest, BinaryOperationsTest) {
   std::vector<T> data1 = {1.0, 2.0, 3.0};
   std::vector<T> data2 = {4.0, 5.0, 6.0};
   std::vector<size_t> shape = {3}; // 3×1 tensors.
   Tensor<T> t1 = create_tensor(data1, shape);
   Tensor<T> t2 = create_tensor(data2, shape);

   // Addition
   Tensor<T> addResult = t1 + t2;
   Tensor<T> expectedAdd = create_tensor(std::vector<T>{5.0, 7.0, 9.0}, shape);
   for (size_t i = 0; i < expectedAdd.size(); ++i) {
      EXPECT_NEAR(addResult[i], expectedAdd[i], 1e-6);
   }

   // Subtraction
   Tensor<T> subResult = t2 - t1;
   Tensor<T> expectedSub = create_tensor(std::vector<T>{3.0, 3.0, 3.0}, shape);
   ;
   for (size_t i = 0; i < expectedSub.size(); ++i) {
      EXPECT_NEAR(subResult[i], expectedSub[i], 1e-6);
   }
   //
   // Elementwise Mul
   Tensor<T> mulResult = t1 * t2;
   Tensor<T> expectedMul =
       create_tensor(std::vector<T>{4.0, 10.0, 18.0}, shape);
   ;
   for (size_t i = 0; i < expectedMul.size(); ++i) {
      EXPECT_NEAR(mulResult[i], expectedMul[i], 1e-6);
   }

   // Elementwise Division
   Tensor<T> divResult = t2 / t1;
   Tensor<T> expectedDiv = create_tensor(std::vector<T>{4.0, 2.5, 2.0}, shape);
   for (size_t i = 0; i < expectedDiv.size(); ++i) {
      EXPECT_NEAR(divResult[i], expectedDiv[i], 1e-6);
   }

   // Pow: Note that Tensor::pow takes a scalar exponent.
   // Here, we raise each element of t1 to the 4th power.
   Tensor<T> tensor_to_pow = create_tensor(std::vector<T>{4.0}, {1});
   Tensor<T> powResult = t1.pow(tensor_to_pow);
   std::vector<T> expectedPow;
   for (T d : data1) {
      expectedPow.push_back(std::pow(d, 4.0));
   }
   for (size_t i = 0; i < expectedPow.size(); ++i) {
      EXPECT_NEAR(powResult[i], expectedPow[i], 1e-6);
   }
}

//------------------------------------------------------------------------------
// Binary Operation With "Scalar"
// Since our Tensor class does not support direct scalar–tensor operations,
// we simulate a scalar by creating a tensor of matching shape with all
// elements
// equal.
TEST(TensorOpsTest, BinaryOpWithScalarTest) {
   std::vector<T> data = {2.0, 4.0, 6.0};
   std::vector<size_t> shape = {3}; // 3×1 tensor.
   Tensor<T> tensor = create_tensor(data, shape);

   // Create a tensor (of same shape) filled with 2.0.
   std::vector<T> scalarData(shape[0], 2.0);
   Tensor<T> scalarTensor = create_tensor(scalarData, shape);

   // Test addition: scalar + tensor.
   Tensor<T> addResult = scalarTensor + tensor;
   std::vector<T> expectedAdd;
   for (T d : data) {
      expectedAdd.push_back(2.0 + d);
   }
   Tensor<T> expectedAddTensor = create_tensor(expectedAdd, shape);
   for (size_t i = 0; i < expectedAddTensor.size(); ++i) {
      EXPECT_NEAR(addResult[i], expectedAddTensor[i], 1e-6);
   }

   // Test subtraction: tensor - scalar.
   Tensor<T> subResult = tensor - scalarTensor;
   std::vector<T> expectedSub;
   for (T d : data) {
      expectedSub.push_back(d - 2.0);
   }
   Tensor<T> expectedSubTensor = create_tensor(expectedSub, shape);
   for (size_t i = 0; i < expectedSub.size(); ++i) {
      EXPECT_NEAR(subResult[i], expectedSubTensor[i], 1e-6);
   }
}

//------------------------------------------------------------------------------
// Mismatched Sizes Test

TEST(TensorOpsTest, MismatchedSizeTest) {
   std::vector<T> data1 = {1.0, 2.0};
   std::vector<T> data2 = {3.0, 4.0, 5.0};
   std::vector<size_t> shape1 = {2}; // 2×1 tensor.
   std::vector<size_t> shape2 = {3}; // 3×1 tensor.
   Tensor<T> t1 = create_tensor(data1, shape1);
   Tensor<T> t2 = create_tensor(data2, shape2);

   EXPECT_THROW(
       {
          auto result = t1 + t2;
          (void)result;
       },
       std::runtime_error);
}

//------------------------------------------------------------------------------
// Matrix Multiplication Tests

TEST(TensorOpsTest, MatrixMultiplication2DTest) {
   // Define two 2x2 matrices:
   // A = [1, 2;
   //      3, 4]
   // B = [5, 6;
   //      7, 8]
   // Expected: A.matmul(B) = [19, 22;
   //                           43, 50]
   std::vector<T> dataA = {1, 2, 3, 4};
   std::vector<size_t> shapeA = {2, 2};
   Tensor<T> A = create_tensor(dataA, shapeA);

   std::vector<T> dataB = {5, 6, 7, 8};
   std::vector<size_t> shapeB = {2, 2};
   Tensor<T> B = create_tensor(dataB, shapeB);

   Tensor<T> result = A.matmul(B);
   std::vector<T> expected = {19, 22, 43, 50};
   Tensor<T> expectedResult = create_tensor(expected, shapeA);
   ASSERT_EQ(result.size(), expected.size());
   for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(result[i], expected[i], 1e-6);
   }
}

TEST(TensorOpsTest, MatrixMultiplication1DTest) {
   // For a 2x3 matrix A and a 3×1 tensor v:
   // A = [1, 2, 3;
   //      4, 5, 6]
   // v = [1; 1; 1]  (represented as a 3×1 tensor)
   // Expected: A.matmul(v) = [6; 15]
   std::vector<T> dataA = {1, 2, 3, 4, 5, 6};
   std::vector<size_t> shapeA = {2, 3};
   Tensor<T> A = create_tensor(dataA, shapeA);

   std::vector<T> dataV = {1, 1, 1};
   // Using a 1D shape for v creates a 3×1 tensor.
   std::vector<size_t> shapeV = {3};
   Tensor<T> v = create_tensor(dataV, shapeV);

   Tensor<T> result = A.matmul(v);
   std::vector<T> expected = {6, 15};
   Tensor<T> expectedResult = create_tensor(expected, shapeA);
   ASSERT_EQ(result.size(), expected.size());
   for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(result[i], expected[i], 1e-6);
   }
}

TEST(TensorOpsTest, MatrixMultiplicationInvalidShapeTest) {
   std::vector<T> data = {1, 2, 3};
   std::vector<size_t> shape = {3}; // 3×1 tensor.
   Tensor<T> tensor = create_tensor(data, shape);

   EXPECT_THROW(
       {
          auto result = tensor.matmul(tensor);
          (void)result;
       },
       std::runtime_error);
}

//------------------------------------------------------------------------------
// main() for Google Test

int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
