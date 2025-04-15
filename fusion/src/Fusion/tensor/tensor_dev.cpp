#include "eigen_tensor.h"
#include "storage_tensor.h"
#include <Eigen/Dense>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>

template <typename T> class Tensor {
public:
  std::unique_ptr<ITensorStorage<T>> storage;

  explicit Tensor(size_t rows, size_t cols, Device device = Device::CPU) {
    if (device == Device::CPU) {
      storage = std::make_unique<EigenTensorStorage<T>>(rows, cols);
    } else {
      throw std::invalid_argument(
          "Unsupported device type - currently only supports CPU");
    }
  }

  explicit Tensor(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
                                      Eigen::RowMajor> &matrix) {
    storage =
        std::make_unique<EigenTensorStorage<T>>(matrix.rows(), matrix.cols());
    static_cast<EigenTensorStorage<T> *>(storage.get())->matrix = matrix;
  }

  void setValues(std::initializer_list<T> values) {
    size_t expected = storage->rows() * storage->cols();
    if (values.size() != expected) {
      throw std::invalid_argument(
          "Incorrect number of values provided to setValues");
    }
    auto cpuStorage = dynamic_cast<EigenTensorStorage<T> *>(storage.get());
    if (!cpuStorage) {
      throw std::runtime_error("Unsupported storage type in setValues");
    }
    size_t idx = 0;
    for (const auto &val : values) {
      // Since we're using row-major ordering,
      // the element at index idx corresponds to row = idx/cols, col = idx%cols.
      cpuStorage->matrix(idx / storage->cols(), idx % storage->cols()) = val;
      ++idx;
    }
  }

  void setValues(std::initializer_list<std::initializer_list<T>> nestedValues) {
    if (nestedValues.size() != storage->rows()) {
      throw std::invalid_argument("Row count mismatch in nested setValues");
    }
    auto cpuStorage = dynamic_cast<EigenTensorStorage<T> *>(storage.get());
    if (!cpuStorage) {
      throw std::runtime_error(
          "Unsupported storage type in setValues (nested)");
    }
    size_t row = 0;
    for (const auto &rowList : nestedValues) {
      if (rowList.size() != storage->cols()) {
        throw std::invalid_argument(
            "Column count mismatch in nested setValues");
      }
      size_t col = 0;
      for (const auto &val : rowList) {
        cpuStorage->matrix(row, col) = val;
        ++col;
      }
      ++row;
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    const auto *cpuStorage =
        dynamic_cast<const EigenTensorStorage<T> *>(tensor.storage.get());
    if (cpuStorage) {
      os << "Tensor(" << std::endl << cpuStorage->matrix << std::endl << ")";
    } else {
      os << "Tensor(unsupported storage type)";
    }
    return os;
  }

  // overload the + operator
  Tensor<T> operator+(const Tensor<T> &other) const {
    if (this->storage->rows() != other.storage->rows() ||
        this->storage->cols() != other.storage->cols() ||
        this->storage->device() != other.storage->device()) {
      throw std::invalid_argument(
          "Tensor shape or device mismatch in addition");
    }
    const auto *cpuThis =
        dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    const auto *cpuOther =
        dynamic_cast<const EigenTensorStorage<T> *>(other.storage.get());
    if (!cpuThis || !cpuOther) {
      throw std::invalid_argument("Unsupported storage type for addition");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMat = cpuThis->matrix + cpuOther->matrix;
    return Tensor<T>(resultMat);
  }

  // overload the - operator
  Tensor<T> operator-(const Tensor<T> &other) const {
    if (this->storage->rows() != other.storage->rows() ||
        this->storage->cols() != other.storage->cols() ||
        this->storage->device() != other.storage->device()) {
      throw std::invalid_argument(
          "Tensor shape or device mismatch in addition");
    }
    const auto *cpuThis =
        dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    const auto *cpuOther =
        dynamic_cast<const EigenTensorStorage<T> *>(other.storage.get());
    if (!cpuThis || !cpuOther) {
      throw std::invalid_argument("Unsupported storage type for addition");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMat = cpuThis->matrix - cpuOther->matrix;
    return Tensor<T>(resultMat);
  }

  Tensor<T> operator/(const Tensor<T> &other) const {
    if (this->storage->rows() != other.storage->rows() ||
        this->storage->cols() != other.storage->cols() ||
        this->storage->device() != other.storage->device()) {
      throw std::invalid_argument(
          "Tensor shape or device mismatch in division");
    }
    const auto *cpuThis =
        dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    const auto *cpuOther =
        dynamic_cast<const EigenTensorStorage<T> *>(other.storage.get());
    if (!cpuThis || !cpuOther) {
      throw std::invalid_argument("Unsupported storage type for division");
    }
    // Use the array() method to perform elementwise division.
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMat =
            (cpuThis->matrix.array() / cpuOther->matrix.array()).matrix();
    return Tensor<T>(resultMat);
  }

  // overload the + operator
  Tensor<T> operator*(const Tensor<T> &other) const {
    if (this->storage->rows() != other.storage->rows() ||
        this->storage->cols() != other.storage->cols() ||
        this->storage->device() != other.storage->device()) {
      throw std::invalid_argument(
          "Tensor shape or device mismatch in addition");
    }
    const auto *cpuThis =
        dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    const auto *cpuOther =
        dynamic_cast<const EigenTensorStorage<T> *>(other.storage.get());
    if (!cpuThis || !cpuOther) {
      throw std::invalid_argument("Unsupported storage type for addition");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMat = cpuThis->matrix * cpuOther->matrix;
    return Tensor<T>(resultMat);
  }

  Tensor<T> sqrt() const {
    const auto *cpuThis =
        dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    if (!cpuThis) {
      throw std::runtime_error("Unsupported storage type for sqrt");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMat = cpuThis->matrix.array().sqrt().matrix();
    return Tensor<T>(resultMat);
  }

  Tensor<T> pow(const T exponent) const {
    const auto *cpuThis =
        dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    if (!cpuThis) {
      throw std::runtime_error("Unsupported storage type for pow");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        resultMat = cpuThis->matrix.array().pow(exponent).matrix();
    return Tensor<T>(resultMat);
  }
};

int main() {
  Tensor<double> tensorA(2, 2, Device::CPU);
  Tensor<double> tensorB(2, 2, Device::CPU);

  tensorA.setValues({1, 2, 3, 4});
  tensorB.setValues({5, 6, 7, 8});

  Tensor<double> tensorC = tensorA / tensorB;

  std::cout << "tensorA:\n" << tensorA << std::endl;
  std::cout << "tensorB:\n" << tensorB << std::endl;
  std::cout << "tensorA + tensorB:\n" << tensorC << std::endl;

  return 0;
}
