#include "eigen_tensor.h"
#include "storage_tensor.h"
#include <iostream>
#include <memory>
#include <stdexcept>

template <typename T>
class Tensor { // TODO: evaluate the memory overhead of this approach...
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

  friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
    const EigenTensorStorage<T> *cpuStorage =
        dynamic_cast<const EigenTensorStorage<T> *>(tensor.storage.get());
    if (cpuStorage) {
      os << "Tensor(" << std::endl << cpuStorage->matrix << std::endl << ")";
    } else {
      os << "Tensor(unsupported storage type)";
    }
    return os;
  }

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
};

int main() {
  Tensor<double> tensorA(2, 2, Device::CPU);
  Tensor<double> tensorB(2, 2, Device::CPU);

  auto storageA =
      dynamic_cast<EigenTensorStorage<double> *>(tensorA.storage.get());
  auto storageB =
      dynamic_cast<EigenTensorStorage<double> *>(tensorB.storage.get());
  if (!storageA || !storageB) {
    std::cerr << "Error: dynamic cast failed!" << std::endl;
    return 1;
  }

  storageA->matrix << 1, 2, 3, 4;

  storageB->matrix << 5, 6, 7, 8;

  const Tensor<double> tensorC = tensorA + tensorB;

  std::cout << "tensorA:\n" << tensorA << std::endl;
  std::cout << "tensorB:\n" << tensorB << std::endl;
  std::cout << "tensorA + tensorB:\n" << tensorC << std::endl;

  return 0;
}
