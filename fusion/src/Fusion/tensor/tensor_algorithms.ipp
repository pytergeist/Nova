#ifndef TENSOR_ALGORITHMS_IPP
#define TENSOR_ALGORITHMS_IPP

template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T> &other) const {
    if (this->storage->cols() != other.storage->rows()) {
        throw std::invalid_argument("Matrix dimension mismatch for matmul");
    }
    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    const auto *cpuOther =
            dynamic_cast<const EigenTensorStorage<T> *>(other.storage.get());
    if (!cpuThis || !cpuOther) {
        throw std::invalid_argument("Unsupported storage type for matmul");
    }
    int m = static_cast<int>(cpuThis->matrix.rows());
    int k = static_cast<int>(cpuThis->matrix.cols());
    int n = static_cast<int>(cpuOther->matrix.cols());

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> resultMat(
        m, n);

    const double alpha = 1.0;
    const double beta = 0.0;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
                cpuThis->matrix.data(), k, cpuOther->matrix.data(), n, beta,
                resultMat.data(), n);

    return Tensor<T>(resultMat);
}

template <typename T>
Tensor<T> Tensor<T>::transpose() const {
    const auto *cpuThis =
            dynamic_cast<const EigenTensorStorage<T> *>(this->storage.get());
    if (!cpuThis) {
        throw std::runtime_error("Unsupported storage type for transpose");
    }
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            transposedMat = cpuThis->matrix.transpose();
    return Tensor<T>(transposedMat);
}

#endif // TENSOR_ALGORITHMS_IPP
