#ifndef DENSE_STORAGE_H
#define DENSE_STORAGE_H

#include <algorithm>
#include "StorageInterface.h"
#include "TensorBuffer.h"

template <typename T> class NDTensorStorage : public ITensorStorage<T> {
private:
  std::vector<size_t> shape_, strides_;
  TensorBuffer data_;

public: // TODO: Be careful here - do we want this ptr to be mutable?
  explicit NDTensorStorage(std::vector<size_t> shape, std::vector<T> data)
      : shape_(std::move(shape)), data_(copy_data_to_buff(data)) {
    size_t sz = 1;
    strides_.resize(shape_.size());
    for (size_t i = 0; i < shape_.size() - 1; i++) {
      strides_[i] = sz;
      sz *= shape_[i];
    }
  };

  TensorBuffer copy_data_to_buff(std::vector<T> vec) {
    TensorBuffer buff = TensorBuffer::allocate_elements<T>(vec.size());
    buff.data_as<T>();
    buff.copy_from<T>(vec, 0); // TODO: dynamically calculate the byte offset
    return buff;
  }

  TensorBuffer &data() override { return data_; }
  const TensorBuffer &data() const override { return data_; }
  T *data_ptr() override { return data_.data<T>(); }
  const T *data_ptr() const override { return data_.data<T>(); }

  [[nodiscard]] std::vector<size_t> shape() const override { return shape_; }
  [[nodiscard]] std::vector<size_t> strides() const override {
    return strides_;
  }
  [[nodiscard]] size_t size() const override { return data_.size<T>(); }
  [[nodiscard]] size_t ndims() const override {
    return std::count_if(shape_.begin(), shape_.end(),
                         [](size_t d) { return d != 1; });
  }

  [[nodiscard]] Device device() const override { return Device::CPU; }
};

#endif // DENSE_STORAGE_H
