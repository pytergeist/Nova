#include "tensor.h"

int main() {
  Tensor<double> const tensor1({1.0, 2.0, 3, 4}, {2, 2});
  Tensor<double> const tensor2({1.0, 2.0, 3, 4}, {2, 2});

  Tensor<double> tensor3 = tensor1.matmul(tensor2);
  std::cout << tensor3 << std::endl;

  return 0;
}
