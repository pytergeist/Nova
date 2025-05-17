#include "tensor/tensor.h"

int main() {

  Tensor<double> t1({2, 2}, {1, 2, 3, 4});
  Tensor<double> t2({2, 2}, {2, 2, 2, 2});
  Tensor<double> t4({2, 2}, {0, 0, 0, 0});

  // Tensor<double> t3 = t1 * t1;
  Tensor<double> t3 = t1.transpose();
  std::cout << t3 << std::endl;
  std::cout << t3.rank_ << std::endl;

  return 0;
};
