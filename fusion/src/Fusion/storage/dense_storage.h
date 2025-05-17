#ifndef DENSE_STORAGE_H
#define DENSE_STORAGE_H

#include "../storage/storage_interface.h"
#include <utility>

template <typename T> class NDTensorStorage : public ITensorStorage<T> {
private:
  std::vector<size_t> shape_, strides_;
  std::vector<T> data_;

public:
  explicit NDTensorStorage(std::vector<size_t> shape, std::vector<T> data)
      : shape_(std::move(shape)), data_(std::move(data)) {
    size_t sz = 1;
    strides_.resize(shape_.size());
    for (size_t i = 0; i < shape_.size() - 1; i++) {
      strides_[i] = sz;
      sz *= shape_[i];
    }
  };

  std::vector<T> &data() override { return data_; }
  const std::vector<T> &data() const override { return data_; }

  [[nodiscard]] std::vector<size_t> shape() const override { return shape_; }
  [[nodiscard]] std::vector<size_t> strides() const override {
    return strides_;
  }
  [[nodiscard]] size_t size() const override { return data_.size(); }
  [[nodiscard]] size_t ndims() const override {
    return std::count_if(shape_.begin(), shape_.end(),
                         [](size_t d) { return d != 1; });
  }

  [[nodiscard]] Device device() const override { return Device::CPU; }
};

#endif // DENSE_STORAGE_H
