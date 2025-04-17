#ifndef TENSOR_REDUCTIONS_H
#define TENSOR_REDUCTIONS_H

template<typename T>
Tensor<T> Tensor<T>::sum() const {
    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    if (!cpuThis) {
        throw std::runtime_error("Unsupported storage type for sum");
    }
    T total = cpuThis->matrix.sum();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> totalMat(
        1, 1);
    totalMat(0, 0) = total;
    return Tensor<T>(totalMat);
}

template <typename T>
Tensor<T> Tensor<T>::maximum(const Tensor<T> &other) const {
    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    const auto *cpuOther =
            dynamic_cast<const EigenTensorStorage<T> *>(other.storage.get());
    if (!cpuThis || !cpuOther) {
        throw std::invalid_argument("Unsupported storage type for maximum");
    }

    if (other.storage->rows() == 1 && other.storage->cols() == 1) {
        T threshold = cpuOther->matrix(0, 0);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                resultMat = (cpuThis->matrix.array() >= threshold)
                        .select(cpuThis->matrix.array(), T(0))
                        .matrix();
        return Tensor<T>(resultMat);
    }

    if (this->storage->rows() == other.storage->rows() &&
        this->storage->cols() == other.storage->cols()) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
                resultMat = cpuThis->matrix.cwiseMax(cpuOther->matrix);
        return Tensor<T>(resultMat);
        }

    throw std::invalid_argument("Tensor shapes do not match for maximum");
}

#endif // TENSOR_REDUCTIONS_H
