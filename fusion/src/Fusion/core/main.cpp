#include "tensor.h"

int main() {
  Tensor<double> const tensor1({1.0, 2.0, 3.0, 4.0, 5.0});
  Tensor<double> const tensor2({1.0, 2.0, 3.0, 4.0, 5.0});
  Tensor<double> const &tensor3 = tensor1.pow(tensor2);

  std::cout << tensor3 << std::endl;
  return 0;
}
