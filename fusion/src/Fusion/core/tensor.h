#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

template <typename> struct is_std_vector : std::false_type {};

template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

template <typename T> class Tensor {
public:
  std::vector<T> arr;

  explicit Tensor(const std::vector<T> &data);

  // explicit Tensor(const std::vector<std::vector<T>> &data);

  explicit Tensor(const T &value);

  // friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
  //   os << "Tensor(";
  //   for (size_t i = 0; i < tensor.arr.size(); ++i) {
  //     os << tensor.arr[i];
  //     if (i < tensor.arr.size() - 1)
  //       os << ", ";
  //   }
  //   os << ")";
  //   return os;
  // }

  friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(";
    os << std::endl;
    if constexpr (is_std_vector<T>::value) {
      for (size_t i = 0; i < tensor.arr.size(); ++i) {
        for (size_t j = 0; j < tensor.arr[i].size(); ++j) {
          os << tensor.arr[i][j];
          if (j < tensor.arr[i].size() - 1)
            os << ", ";
        }
        os << std::endl;
      }
      os << ")";
      return os;
    }
    if constexpr (!is_std_vector<T>::value) {
      for (size_t i = 0; i < tensor.arr.size(); ++i) {
        os << tensor.arr[i];
        if (i < tensor.arr.size() - 1)
          os << ", ";
      }
      os << ")";
      return os;
    }
  }

  // binary operations
  Tensor<T> operator+(const Tensor<T> &tensor) const;

  Tensor<T> operator-(const Tensor<T> &tensor) const;

  Tensor<T> operator*(const Tensor<T> &tensor) const;

  Tensor<T> operator/(const Tensor<T> &tensor) const;

  Tensor<T> pow(const Tensor<T> &tensor) const;

  // unary operations
  Tensor<T> sqrt() const;

  Tensor<T> exp() const;

  Tensor<T> log() const;
};

#include "tensor_ops.tpp"
#endif // TENSOR_H
