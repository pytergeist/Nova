#ifndef TENSOR_CONSTRUCTORS_IPP
#define TENSOR_CONSTRUCTORS_IPP
#include <vector>


template<typename T>
Tensor<T>::Tensor(size_t rows, size_t cols, Device device) {
    if (device == Device::CPU) {
        storage = std::make_unique<EigenTensorStorage<T> >(rows, cols);
        shape_ = {rows, cols};
    } else {
        throw std::invalid_argument("Unsupported device type");
    }
}

template<typename T>
Tensor<T>::Tensor(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic,
    Eigen::RowMajor> &matrix) {
    storage =
            std::make_unique<EigenTensorStorage<T> >(matrix.rows(), matrix.cols());
    shape_ = {
        static_cast<size_t>(matrix.rows()),
        static_cast<size_t>(matrix.cols())
    };
    static_cast<EigenTensorStorage<T> *>(storage.get())->matrix = matrix;
}

template<typename T>
Tensor<T>::Tensor(T value, Device device) {
    if (device == Device::CPU) {
        storage = std::make_unique<EigenTensorStorage<T> >(1, 1);
        shape_ = {};
        static_cast<EigenTensorStorage<T> *>(storage.get())->matrix(0, 0) = value;
    } else {
        throw std::invalid_argument("Unsupported device type");
    }
}

template<typename T>
std::vector<size_t> Tensor<T>::shape() const { return this->shape_; }


template<typename T>
void Tensor<T>::setValues(std::initializer_list<T> values) {
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
    for (const auto &val: values) {
        cpuStorage->matrix(idx / storage->cols(), idx % storage->cols()) = val;
        ++idx;
    }
}


template<typename T>
void Tensor<T>::setValues(std::initializer_list<std::initializer_list<T> > nestedValues) {
    if (nestedValues.size() != storage->rows()) {
        throw std::invalid_argument("Row count mismatch in nested setValues");
    }
    auto cpuStorage = dynamic_cast<EigenTensorStorage<T> *>(storage.get());
    if (!cpuStorage) {
        throw std::runtime_error(
            "Unsupported storage type in setValues (nested)");
    }
    size_t row = 0;
    for (const auto &rowList: nestedValues) {
        if (rowList.size() != storage->cols()) {
            throw std::invalid_argument(
                "Column count mismatch in nested setValues");
        }
        size_t col = 0;
        for (const auto &val: rowList) {
            cpuStorage->matrix(row, col) = val;
            ++col;
        }
        ++row;
    }
}


#endif // TENSOR_CONSTRUCTORS_IPP
