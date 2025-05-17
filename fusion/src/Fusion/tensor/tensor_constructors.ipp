//#ifndef TENSOR_CONSTRUCTORS_IPP
//#define TENSOR_CONSTRUCTORS_IPP
//#include <vector>
//#include <memory>
//
//
//template<typename T>
//Tensor<T>::Tensor(std::vector<size_t> shape, Device device) {
//    if (device == Device::CPU) {
//        storage = std::make_unique<NDTensorStorage<T> >(shape);
//    } else {
//        throw std::invalid_argument("Unsupported device type");
//    }
//}
//
//template<typename T>
//Tensor<T>::Tensor(T value, Device device) {
//    if (device == Device::CPU) {
//        storage = std::make_unique<EigenTensorStorage<T> >(1, 1);
//        shape_ = {};
//        static_cast<EigenTensorStorage<T> *>(storage.get())->matrix(0, 0) = value;
//    } else {
//        throw std::invalid_argument("Unsupported device type");
//    }
//}
//
//
//#endif // TENSOR_CONSTRUCTORS_IPP
