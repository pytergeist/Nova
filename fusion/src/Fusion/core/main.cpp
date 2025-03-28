#include "tensor.h"

int main() {
  Tensor<double> const tensor1({1.0, 2.0, 3, 4}, {2, 2});
  Tensor<double> const tensor2({1.0, 2.0, 3, 4}, {2, 2});
  Tensor<double> const tensor3({1.0, 2.0}, {2});

  Tensor<double> tensor4 = tensor1.matmul(tensor3);
  std::cout << tensor4 << std::endl;

  return 0;
}
