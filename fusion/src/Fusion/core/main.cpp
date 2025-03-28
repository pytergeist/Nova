#include "tensor.h"

int main() {
  Tensor<double> const tensor1({2.0, 2.0, 3, 4}, {2,2});
  Tensor<double> const tensor2({2.0, 2.0, 3, 4}, {1});




  Tensor<double> plus_tensor = tensor2+tensor2;
  Tensor<double> minus_tensor = tensor2-tensor2;
  Tensor<double> divide_tensor = tensor2/tensor2;
  Tensor<double> times_tensor = tensor2*tensor2;
  Tensor<double> pow_tensor = tensor2.pow(tensor2);
  Tensor<double> sqrt_tensor = tensor2.sqrt();
  Tensor<double> exp_tensor = tensor2.exp();
  Tensor<double> log_tensor = tensor2.log();


  std::cout << tensor1 << std::endl;
  std::cout << tensor2 << std::endl;
  std::cout << plus_tensor << std::endl;
  std::cout << minus_tensor << std::endl;
  std::cout << times_tensor << std::endl;
  std::cout << pow_tensor << std::endl;
  std::cout << times_tensor << std::endl;
  std::cout << sqrt_tensor << std::endl;
  std::cout << exp_tensor << std::endl;
  std::cout << log_tensor << std::endl;



  return 0;
}
