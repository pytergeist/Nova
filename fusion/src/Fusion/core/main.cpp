#include "tensor.h"

int main() {
  Tensor<std::vector<double>> const tensor1({{1.0, 2.0}, {3.0, 4.0}});

  Tensor<double> const tensor2({2.0, 2.0});
  Tensor<std::vector<double>> const tensor3({{1.0, 2.0}, {3.0, 4.0}});

  Tensor<std::vector<double>> const tensor4 = tensor1.matmul(tensor3);

  // size_t const rows = tensor1.arr.size();
  // constexpr int cols = 1;
  // std::vector<std::vector<double>> result;
  // result.resize(rows);
  // for (size_t i = 0; i < rows; i++) {
  //     result[i].resize(cols);
  //     for (size_t j = 0; j < cols; j++) {
  //         result[i][j] = 0;
  //     }
  // }
  //
  // for (size_t i = 0; i < rows; i++) {
  //     for (size_t j = 0; j < cols; j++) {
  //         for (size_t k = 0; k < tensor1.arr[i].size(); k++) {
  //             result[i][j] += tensor1.arr[i][k]*tensor2.arr[j];
  //         }
  //     }
  // }
  //
  // size_t const rows = tensor1.arr.size();
  // size_t const cols = tensor1.arr[0].size();
  // std::vector<std::vector<double>> result;
  // result.resize(rows);
  // for (size_t i = 0; i < rows; i++) {
  //     result[i].resize(cols);
  //     for (size_t j = 0; j < cols; j++) {
  //       result[i][j] = 0;
  //     }
  // }
  //
  // for (size_t i = 0; i < rows; i++) {
  //     for (size_t j = 0; j < cols; j++) {
  //         for (size_t k = 0; k < tensor1.arr[i].size(); k++) {
  //             result[i][j] += tensor1.arr[i][k]*tensor2.arr[k][j];
  //         }
  //     }
  // }
  // Tensor<std::vector<double>> const tensor3(result);

  std::cout << tensor4 << std::endl;
  return 0;
}
