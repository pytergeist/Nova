#include "tensor.h"

int main() {

  std::vector<double> v1{1, 2, 3, 4};
  std::vector<double> v2{1};

  Tensor<double> t1({2, 2}, v1);
  Tensor<double> t2({1}, v2);

  // Tensor<double> t3 = t1 * t1;
  const Tensor<double> t3 = t1 + t2;
  std::cout << t3 << std::endl;
  std::cout << t3.rank_ << std::endl;
  std::cout << t1 << std::endl;
  std::cout << t1.rank_ << std::endl;
  const Tensor<double> t4 = t2 + t1;
  std::cout << t4 << std::endl;
  std::cout << t4.rank_ << std::endl;

  return 0;
};
