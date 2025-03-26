#include "tensor.h"

int main() {
  Tensor<double> const tensor1({1.0, 2.0, 3.0, 4.0, 5.0});
  Tensor<double> const tensor2({1.0, 2.0, 3.0, 4.0, 5.0});
  Tensor<double> const &tensor3 = tensor2.log();

  Tensor<std::vector<double>> const tensor4(
      {{1.0, 2.0, 4.0, 8.0, 6.0}, {1.0, 2.0, 3.0, 4.0, 5.0}});

  std::cout << tensor4 << std::endl;
  return 0;
}
