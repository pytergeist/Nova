#ifndef DENSE_STORAGE
#define EIGEN_TENSOR_H

#include "../storage/storage_interface.h"
#include <Eigen/Dense>

template <typename T> class NDTensorStorage : public ITensorStorage<T> {
private:
  std::vector<size_t> shape_, strides_;
  std::vector<T> data_;
public:
    NDTensorStorage(std::vector<size_t> shape) : shape_(shape) {
      size_t sz = 1;
      strides_.resize(shape_.size());
      for (size_t i = 0; i < shape_.size() - 1; i++) {
        strides_[i] = sz;
        sz *= shape_[i];
      }
    };

    T *data() override { return data_.data(); }
    const T *data() const override { return data_.data(); }

    std::vector<size_t> shape() const override { return shape_; }
    std::vector<size_t> strides() const override { return strides_; }
    std::vector<size_t> size() const override { return data_.size(); }
    size_t ndims() const override {
        return std::count_if(
          shape_.begin(), shape_.end(),
          [](size_t d){ return d != 1; }
        );
      }

    Device device() const override { return Device::CPU; }
};

#endif // EIGEN_TENSOR_H
