#include "../tensor/tensor.h"
#include <cmath>
#include <gtest/gtest.h>
#include <stdexcept>

// ---------- Unary Operations Tests ----------

TEST(TensorOpsTest, UnarySqrtTest) {
  const std::vector<double> data = {4.0, 9.0, 16.0};
  const std::vector<size_t> shape = {3};
  const Tensor<double> tensor(data, shape);
  const Tensor<double> result = tensor.sqrt();

  std::vector<double> expected = {2.0, 3.0, 4.0};
  ASSERT_EQ(result.arr.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result.arr[i], expected[i], 1e-6);
  }
}

TEST(TensorOpsTest, UnaryExpTest) {
  const std::vector<double> data = {0.0, 1.0};
  const std::vector<size_t> shape = {2};
  const Tensor<double> tensor(data, shape);
  const Tensor<double> result = tensor.exp();

  const std::vector<double> expected = {std::exp(0.0), std::exp(1.0)};
  ASSERT_EQ(result.arr.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result.arr[i], expected[i], 1e-6);
  }
}

// Test elementwise log: tensor.log() should return the natural logarithms.
TEST(TensorOpsTest, UnaryLogTest) {
  const std::vector<double> data = {std::exp(1.0), std::exp(2.0)};
  const std::vector<size_t> shape = {2};
  const Tensor<double> tensor(data, shape);
  const Tensor<double> result = tensor.log();

  std::vector<double> expected = {1.0, 2.0};
  ASSERT_EQ(result.arr.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result.arr[i], expected[i], 1e-6);
  }
}

// ---------- Binary Operations Tests ----------

// Test elementwise binary operations: addition, subtraction, multiplication,
// division, and pow.
TEST(TensorOpsTest, BinaryOperationsTest) {
  std::vector<double> data1 = {1.0, 2.0, 3.0};
  std::vector<double> data2 = {4.0, 5.0, 6.0};
  std::vector<size_t> shape = {3};
  Tensor<double> t1(data1, shape);
  Tensor<double> t2(data2, shape);

  // Addition
  Tensor<double> add_result = t1 + t2;
  std::vector<double> expected_add = {5.0, 7.0, 9.0};
  for (size_t i = 0; i < expected_add.size(); ++i) {
    EXPECT_NEAR(add_result.arr[i], expected_add[i], 1e-6);
  }

  // Subtraction
  Tensor<double> sub_result = t2 - t1;
  std::vector<double> expected_sub = {3.0, 3.0, 3.0};
  for (size_t i = 0; i < expected_sub.size(); ++i) {
    EXPECT_NEAR(sub_result.arr[i], expected_sub[i], 1e-6);
  }

  // Multiplication
  Tensor<double> mul_result = t1 * t2;
  std::vector<double> expected_mul = {4.0, 10.0, 18.0};
  for (size_t i = 0; i < expected_mul.size(); ++i) {
    EXPECT_NEAR(mul_result.arr[i], expected_mul[i], 1e-6);
  }

  // Division
  Tensor<double> div_result = t2 / t1;
  std::vector<double> expected_div = {4.0, 2.5, 2.0};
  for (size_t i = 0; i < expected_div.size(); ++i) {
    EXPECT_NEAR(div_result.arr[i], expected_div[i], 1e-6);
  }

  // Pow: t1.pow(t2) computes elementwise power (i.e. 1^4, 2^5, 3^6)
  Tensor<double> pow_result = t1.pow(t2);
  std::vector<double> expected_pow = {std::pow(1.0, 4.0), std::pow(2.0, 5.0),
                                      std::pow(3.0, 6.0)};
  for (size_t i = 0; i < expected_pow.size(); ++i) {
    EXPECT_NEAR(pow_result.arr[i], expected_pow[i], 1e-6);
  }
}

// Test binary operations with one tensor being a scalar.
TEST(TensorOpsTest, BinaryOpWithScalarTest) {
  std::vector<double> data = {2.0, 4.0, 6.0};
  std::vector<size_t> shape = {3};
  Tensor<double> tensor(data, shape);
  Tensor<double> scalar(2.0);

  // Scalar + tensor
  const Tensor<double> add_result = scalar + tensor;
  std::vector<double> expected_add;
  for (double const d : data) {
    expected_add.push_back(2.0 + d);
  }
  for (size_t i = 0; i < expected_add.size(); ++i) {
    EXPECT_NEAR(add_result.arr[i], expected_add[i], 1e-6);
  }

  // Tensor - scalar
  Tensor<double> sub_result = tensor - scalar;
  std::vector<double> expected_sub;
  for (double const d : data) {
    expected_sub.push_back(d - 2.0);
  }
  for (size_t i = 0; i < expected_sub.size(); ++i) {
    EXPECT_NEAR(sub_result.arr[i], expected_sub[i], 1e-6);
  }
}

// Test that operations on tensors with incompatible sizes (non-scalar) throw an
// exception.
TEST(TensorOpsTest, MismatchedSizeTest) {
  std::vector<double> data1 = {1.0, 2.0};
  std::vector<double> data2 = {3.0, 4.0, 5.0};
  std::vector<size_t> shape1 = {2};
  std::vector<size_t> shape2 = {3};
  Tensor<double> t1(data1, shape1);
  Tensor<double> t2(data2, shape2);

  EXPECT_THROW({ auto result = t1 + t2; }, std::invalid_argument);
}

// ---------- Matrix Multiplication Tests ----------

// Test for 2D x 2D matrix multiplication.
// For matrices A (2x2) and B (2x2):
// A = [1, 2; 3, 4], B = [5, 6; 7, 8]
// Expected A.matmul(B) = [19, 22; 43, 50]
TEST(TensorOpsTest, MatrixMultiplication2DTest) {
  std::vector<double> dataA = {1, 2, 3, 4};
  std::vector<size_t> shapeA = {2, 2};
  const Tensor<double> A(dataA, shapeA);

  std::vector<double> dataB = {5, 6, 7, 8};
  std::vector<size_t> shapeB = {2, 2};
  const Tensor<double> B(dataB, shapeB);

  const Tensor<double> result = A.matmul(B);
  const std::vector<double> expected = {19, 22, 43, 50};
  ASSERT_EQ(result.arr.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result.arr[i], expected[i], 1e-6);
  }
}

// Test for 2D x 1D matrix multiplication.
// For a 2x3 matrix A = [1, 2, 3; 4, 5, 6] and vector v = [1, 1, 1],
// Expected A.matmul(v) = [6, 15]
TEST(TensorOpsTest, MatrixMultiplication1DTest) {
  const std::vector<double> dataA = {1, 2, 3, 4, 5, 6};
  const std::vector<size_t> shapeA = {2, 3};
  const Tensor<double> A(dataA, shapeA);

  const std::vector<double> dataV = {1, 1, 1};
  const std::vector<size_t> shapeV = {3};
  const Tensor<double> v(dataV, shapeV);

  const Tensor<double> result = A.matmul(v);
  const std::vector<double> expected = {6, 15};
  ASSERT_EQ(result.arr.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result.arr[i], expected[i], 1e-6);
  }
}

// Test that calling matmul with unsupported shapes (e.g., 1D x 1D) throws an
// exception.
TEST(TensorOpsTest, MatrixMultiplicationInvalidShapeTest) {
  const std::vector<double> data = {1, 2, 3};
  const std::vector<size_t> shape = {3};
  const Tensor<double> tensor(data, shape);

  EXPECT_THROW({ auto result = tensor.matmul(tensor); }, std::invalid_argument);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
