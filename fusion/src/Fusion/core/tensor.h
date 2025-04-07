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
  std::vector<size_t> shape;

  explicit Tensor(const std::vector<T> &data, const std::vector<size_t> &shape);

  explicit Tensor(const T &value);

  static std::ostream &print_2d_tensor(std::ostream &os, const Tensor &tensor) {
    os << std::endl;
    const size_t stride = tensor.shape[1];
    for (size_t i = 0; i < tensor.arr.size(); ++i) {
      os << tensor.arr[i];
      if ((i % stride) != (stride - 1)) {
        os << ", ";
      } else
        os << std::endl;
    }
    os << ")";
    return os;
  }

  static std::ostream &print_1d_tensor(std::ostream &os, const Tensor &tensor) {
    for (size_t i = 0; i < tensor.arr.size(); ++i) {
      os << tensor.arr[i];
      if (i < tensor.arr.size() - 1)
        os << ", ";
    }
    os << ")";
    return os;
  }

  friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    os << "Tensor(";
    if (tensor.shape.size() == 2) {
      print_2d_tensor(os, tensor);
    }
    if (tensor.shape.size() == 1) {
      print_1d_tensor(os, tensor);
    }
    return os;
  }

  // binary operations
  Tensor<T> operator+(const Tensor<T> &tensor) const;

  Tensor<T> operator-(const Tensor<T> &tensor) const;

  Tensor<T> operator*(const Tensor<T> &tensor) const;

  Tensor<T> operator/(const Tensor<T> &tensor) const;

  [[nodiscard]] Tensor<T> pow(const Tensor<T> &tensor) const;

  // unary operations
  [[nodiscard]] Tensor<T> sqrt() const;

  [[nodiscard]] Tensor<T> exp() const;

  [[nodiscard]] Tensor<T> log() const;

  [[nodiscard]] Tensor<T> sum() const;

  // matrix operations
  [[nodiscard]] Tensor<T> matmul(const Tensor<T> &tensor) const;
  [[nodiscard]] Tensor<T> transpose() const;
};

#include "tensor_ops.tpp"
#endif // TENSOR_H
