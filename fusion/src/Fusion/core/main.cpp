#include "tensor.h"

int main() {
  Tensor<double> const tensor1({1.0, 2.0, 3, 4}, {2, 2});
  Tensor<double> const tensor2({1.0, 2.0, 3, 4, 5, 6}, {2, 3});
  Tensor<double> const tensor3({1.0, 2.0}, {2});

  const Tensor<double> tensor4 = tensor1.transpose();
  std::cout << tensor4 << std::endl;

  return 0;
}
