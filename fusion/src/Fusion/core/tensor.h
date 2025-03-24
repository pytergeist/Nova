#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

template <typename T> class Tensor {
public:
  std::vector<T> arr;

  explicit Tensor(const std::vector<T> &data);

  explicit Tensor(const T &value);

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

  Tensor<T> operator+(const Tensor<T> &tensor) const;
  Tensor<T> operator-(const Tensor<T> &tensor) const;
  Tensor<T> operator*(const Tensor<T> &tensor) const;
  Tensor<T> operator/(const Tensor<T> &tensor) const;
  Tensor<T> pow(const Tensor<T> &tensor) const;
  Tensor<T> sqrt() const;
};

#include "tensor_ops.tpp"
#endif // TENSOR_H
