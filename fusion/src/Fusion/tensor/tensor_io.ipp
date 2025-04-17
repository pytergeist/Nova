#ifndef TENSOR_IO_IPP
#define TENSOR_IO_IPP

template <typename T>
std::ostream& operator<<(std::ostream &os, const Tensor<T> &tensor) {
    const auto *cpuStorage =
            dynamic_cast<const EigenTensorStorage<T> *>(tensor.storage.get());
    if (cpuStorage) {
        os << "Tensor(" << std::endl << cpuStorage->matrix << std::endl << ")";
    } else {
        os << "Tensor(unsupported storage type)";
    }
    return os;
}


#endif //TENSOR_IO_IPP
