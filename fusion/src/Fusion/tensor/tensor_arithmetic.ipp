#ifndef TENSOR_ARITHMETIC_IPP
#define TENSOR_ARITHMETIC_IPP

#include "tensor.h"

template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    if (storage->rows() != other.storage->rows() ||
        storage->cols() != other.storage->cols() ||
        storage->device() != other.storage->device()) {
        throw std::invalid_argument(
            "Tensor shape or device mismatch in addition");
        }
    auto cpuThis  = dynamic_cast<const EigenTensorStorage<T>*>(storage.get());
    auto cpuOther = dynamic_cast<const EigenTensorStorage<T>*>(other.storage.get());
    if (!cpuThis || !cpuOther) {
        throw std::invalid_argument("Unsupported storage type for addition");
    }
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>
      resultMat = cpuThis->matrix + cpuOther->matrix;
    return Tensor<T>(resultMat);
}



template<typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other) const {
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

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T> &other) const {
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
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            resultMat =
                    (cpuThis->matrix.array() / cpuOther->matrix.array()).matrix();
    return Tensor<T>(resultMat);
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &other) const {
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
            resultMat =
                    (cpuThis->matrix.array() * cpuOther->matrix.array()).matrix();
    return Tensor<T>(resultMat);
}

template <typename T>
Tensor<T> Tensor<T>::sqrt() const {
    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    if (!cpuThis) {
        throw std::runtime_error("Unsupported storage type for sqrt");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            resultMat = cpuThis->matrix.array().sqrt().matrix();
    return Tensor<T>(resultMat);
}


template <typename T>
Tensor<T> Tensor<T>::exp() const {
    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    if (!cpuThis) {
        throw std::runtime_error("Unsupported storage type for sqrt");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            resultMat = cpuThis->matrix.array().exp().matrix();
    return Tensor<T>(resultMat);
}

template <typename T>
Tensor<T> Tensor<T>::log() const {
    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    if (!cpuThis) {
        throw std::runtime_error("Unsupported storage type for sqrt");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            resultMat = cpuThis->matrix.array().log().matrix();
    return Tensor<T>(resultMat);
}

template <typename T>
Tensor<T> Tensor<T>::pow(const T exponent) const {
    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    if (!cpuThis) {
        throw std::runtime_error("Unsupported storage type for scalar pow");
    }
    // Using Eigen's array::pow method for scalars.
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            resultMat = cpuThis->matrix.array().pow(exponent).matrix();
    return Tensor<T>(resultMat);
}

template <typename T>
Tensor<T> Tensor<T>::pow(const Tensor<T> &exponent) const {
    // Check that the dimensions match.
    if (this->storage->rows() != exponent.storage->rows() ||
        this->storage->cols() != exponent.storage->cols()) {
        throw std::invalid_argument(
            "Tensor shapes must match for elementwise power");
        }

    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    const auto *cpuExponent =
            dynamic_cast<const EigenTensorStorage<T> *>(exponent.storage.get());
    if (!cpuThis || !cpuExponent) {
        throw std::runtime_error("Unsupported storage type for elementwise pow");
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            resultMat =
                    cpuThis->matrix.array()
                    .binaryExpr(cpuExponent->matrix.array(),
                                [](T base, T exp) { return std::pow(base, exp); })
                    .matrix();

    return Tensor<T>(resultMat);
}

#endif // TENSOR_ARITHMETIC_IPP
