#include "tensor.h"

int main() {
  Tensor<float> T1({2, 2}, {1, 2, 3, 4});
  Tensor<float> T2({1}, {2});
  Tensor<float> T3 = T1 + T2;
  std::cout << T3 << std::endl;
  return 0;
};
