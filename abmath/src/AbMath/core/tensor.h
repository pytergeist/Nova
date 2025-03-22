#ifndef TENSOR_H
#define TENSOR_H



#include <iostream>
#include <vector>
#include <stdexcept>

class Tensor {
  public:
    std::vector<double> arr;

    explicit Tensor(const std::vector<double> &data) : arr(data) {}

  friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
      os << "Tensor(";
      for (size_t i = 0; i < tensor.arr.size(); ++i) {
        os << tensor.arr[i];
        if (i < tensor.arr.size() - 1)
          os << ", ";
      }
      os << ")";
      return os;
    }

    Tensor operator+(const Tensor &tensor) const;
    Tensor operator-(const Tensor &tensor) const;
    Tensor operator*(const Tensor &tensor) const;
    Tensor operator/(const Tensor &tensor) const;

};

#endif //TENSOR_H
