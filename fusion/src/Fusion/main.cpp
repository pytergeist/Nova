#include "tensor/tensor.h"

int main() {
  Tensor<double> t1({2, 2}, {2, 2, 2, 2});
  Tensor<double> t2({1}, {1});

  Tensor<double> t3 = t1 * t1;

  std::cout << t3 << std::endl;
  std::cout << t3.rank_ << std::endl;

  return 0;
};
