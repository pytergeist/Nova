#ifndef TENSOR_BASE_HPP
#define TENSOR_BASE_HPP

#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Fusion/alloc/DefaultAllocator.h"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/DType.h"
#include "Fusion/core/ElementWise.hpp"
#include "Fusion/core/Layout.h"
#include "Fusion/core/Reduce.hpp"
#include "Fusion/device/Device.h"
#include "Fusion/kernels/Serial.hpp"
#include "Fusion/ops/Comparison.hpp"
#include "Fusion/ops/Ewise.hpp"
#include "Fusion/ops/Helpers.hpp"
#include "Fusion/ops/Linalg.hpp"
#include "Fusion/ops/OpParams.hpp"
#include "Fusion/ops/Reduce.hpp"
#include "Fusion/ops/Transcendental.hpp"
#include "Fusion/storage/DenseStorage.hpp"
#include "Fusion/storage/StorageInterface.hpp"
#include "Fusion/storage/TensorView.hpp"

template <typename T> // TODO: need to either pass in device somehow?
inline TensorBase<T> scalar_t(const T scalar,
                              const DType dtype = DType::FLOAT32,
                              Device device = Device{DeviceType::CPU, 0}) {
   return TensorBase<T>{{1}, {scalar}, dtype, device};
}

template <typename T> class TensorBase {
 public:
   static constexpr std::string_view name = "TensorBase";
   using value_type = T;

   TensorBase()
       : storage_(nullptr), device_(Device{DeviceType::CPU, 0}), shape_{} {}

   TensorBase(const TensorBase &) = default;
   TensorBase &operator=(const TensorBase &) = default;

   TensorBase(TensorBase &&) noexcept = default;
   TensorBase &operator=(TensorBase &&) noexcept = default;

   ~TensorBase() = default;

   explicit TensorBase(std::vector<std::size_t> shape, std::vector<T> data,
                       DType dtype, Device device,
                       IAllocator *allocator = nullptr)
       : shape_(std::move(shape)), dtype_(dtype), device_(device) {
      FUSION_CHECK(device.is_cpu(), "Unsupported device type");
      FUSION_CHECK(!shape_.empty(), "Tensor: empty shape");
      std::size_t sz = set_contiguous_strides();
      FUSION_CHECK(data.size() == sz, "Tensor: data size != product(shape)");
      auto *alloc = allocator ? allocator : &default_allocator();
      storage_ = make_storage_with_data(shape_, data, device_, alloc);
   }

   explicit TensorBase(std::vector<size_t> shape, DType dtype, Device device,
                       IAllocator *allocator = nullptr)
       : shape_(std::move(shape)), dtype_(dtype), device_(device) {
      FUSION_CHECK(device.is_cpu(), "Unsupported device type");
      FUSION_CHECK(!shape_.empty(), "Tensor: empty shape");
      std::size_t sz = set_contiguous_strides();
      auto *alloc = allocator ? allocator : &default_allocator();
      storage_ = make_storage(shape_, sz, device_, alloc);
   }

   DType dtype() const noexcept { return dtype_; }
   std::size_t dtype_size() const noexcept { return get_dtype_size(dtype_); }

   size_t rank() const { return shape_.size(); }
   size_t ndims() const { return shape_.size(); }
   std::vector<size_t> shape() const { return shape_; }
   std::vector<size_t> strides() const { return strides_; }
   Device device() const noexcept { return device_; }

   bool is_contiguous() const noexcept {
      return calc_contiguous(shape_, strides_);
   }

   std::size_t size() const noexcept {
      return storage_->data().template size<T>();
   }

   bool empty() const noexcept { return !storage_ || storage_->data().empty(); }

   bool is_initialised() const noexcept { return storage_ != nullptr; }
   std::size_t flat_size() const { return storage_->size(); }

   ITensorStorage<T> *get_storage() { return storage_.get(); }
   const ITensorStorage<T> *get_storage() const { return storage_.get(); }

   std::shared_ptr<ITensorStorage<T>> &storage() { return storage_; }
   std::shared_ptr<ITensorStorage<T>> &storage() const { return storage_; }

   TensorBuffer &raw_data() { return storage_->data(); }
   const TensorBuffer &raw_data() const { return storage_->data(); }

   T *get_ptr() { return storage_->data_ptr(); }
   const T *get_ptr() const { return storage_->data_ptr(); }

   TensorView<T>
   view() { // TODO: need to eventuall pass into metadata for views
      return TensorView<T>(storage_->data().template data<T>(), this->shape(),
                           this->strides(), this->rank(), this->ndims());
   }

   T operator[](int idx) const {
      return storage_->data().template data_as<const T>()[idx];
   }

   T *begin() { return storage_->data().template begin<T>(); }
   T *end() { return storage_->data().template end<T>(); }

   T *begin() const { return storage_->data().template begin<T>(); }
   T *end() const { return storage_->data().template end<T>(); }

   std::size_t set_contiguous_strides() {
      size_t sz = 1;
      strides_.resize(shape_.size());
      for (size_t i = 0; i < shape_.size(); i++) {
         strides_[i] = sz;
         sz *= shape_[i];
      };
      return sz;
   }

   void clear() noexcept {
      if (!storage_) {
         return;
      }
      auto &buf = storage_->data();
      if (buf.size_bytes() == 0) {
         return;
      }
      std::memset(buf.data(), 0, buf.size_bytes());
   }

   void assign(const TensorBase &other) {
      if (!storage_) {
         *this = other;
      } else {
         storage_->data().assign(other.begin(), other.end());
      }
   };

   TensorBase operator+(const T scalar) const {
      return fusion::math::add(*this, scalar_t(scalar, dtype(), device()));
   }

   TensorBase operator-(const T scalar) const {
      return fusion::math::sub(*this, scalar_t(scalar, dtype(), device()));
   }

   TensorBase operator*(const T scalar) const {
      return fusion::math::mul(*this, scalar_t(scalar, dtype(), device()));
   }

   TensorBase operator/(const T scalar) const {
      return fusion::math::div(*this, scalar_t(scalar, dtype(), device()));
   }

   TensorBase operator>=(const T scalar) const {
      return fusion::math::greater(*this, scalar_t(scalar, dtype(), device()));
   }

   TensorBase maximum(const T scalar) const {
      return fusion::math::maximum(*this, scalar_t(scalar, dtype(), device()));
   }

   TensorBase pow(const T scalar) const {
      return fusion::math::pow(*this, scalar_t(scalar, dtype(), device()));
   }

   TensorBase operator+(const TensorBase &other) const {
      return fusion::math::add(*this, other);
   }

   TensorBase operator-(const TensorBase &other) const {
      return fusion::math::sub(*this, other);
   }

   TensorBase operator*(const TensorBase &other) const {
      return fusion::math::mul(*this, other);
   }

   TensorBase operator/(const TensorBase &other) const {
      return fusion::math::div(*this, other);
   }

   TensorBase operator>(const TensorBase &other) const {
      return fusion::math::greater(*this, other);
   }

   TensorBase operator>=(const TensorBase &other) const {
      return fusion::math::greater(*this, other);
   }

   TensorBase matmul(const TensorBase &other) const {
      return fusion::math::linalg::matmul(*this, other);
   }

   TensorBase maximum(const TensorBase &other) const {
      return fusion::math::maximum(*this, other);
   }

   TensorBase pow(const TensorBase &other) const {
      return fusion::math::pow(*this, other);
   }

   TensorBase sqrt() const { return fusion::math::sqrt(*this); }
   TensorBase log() const { return fusion::math::log(*this); }
   TensorBase exp() const { return fusion::math::exp(*this); }
   TensorBase sum() const { return fusion::math::sum(*this); }
   TensorBase mean() const { return fusion::math::mean(*this); };

   TensorBase swapaxes(const int axis1, const int axis2) const {
      return fusion::math::linalg::swapaxes(*this, axis1, axis2);
   }

   // TODO: fix this impl -> pipe through ops/kernel layer
   TensorBase transpose() const {
      std::vector<size_t> new_shape(shape_.rbegin(), shape_.rend());

      size_t size = flat_size();
      std::vector<T> new_data(size);

      serial::transpose<T>(*this, this->shape_, new_data);

      return TensorBase(std::move(new_shape), std::move(new_data), dtype(),
                        device_);
   }

   // TODO: fix this impl -> pipe through ops/kernel layer
   TensorBase diagonal() {
      size_t arr_size =
          std::sqrt(std::accumulate(this->shape_.begin(), this->shape_.end(),
                                    int64_t{1}, std::multiplies<int>()));
      size_t out_dim = std::floor(arr_size);
      std::vector<size_t> out_shape{out_dim, 1};
      std::vector<T> out = serial::diagonal2D(*this, this->shape_);
      return TensorBase(std::move(out_shape), std::move(out), dtype(), device_);
   }

   // TODO: fix this impl
   TensorBase &operator-=(const TensorBase &other) {
      BinaryEwiseMeta meta = make_binary_meta(*this, other);
      TensorBase tmp = init_out_from_meta(*this, other, meta);
      ewise::binary_ewise_tag<T, SubtractSIMD>(*this, other, meta, tmp);
      if (!meta.out_shape.empty() && meta.out_shape != tmp.shape()) {
         TensorBase corrected(meta.out_shape, device_, dtype());
         ewise::binary_ewise_tag<T, SubtractSIMD>(*this, other, meta,
                                                  corrected);
         replace_from(std::move(corrected));
      } else {
         replace_from(std::move(tmp));
      }
      return *this;
   }

   friend std::ostream &operator<<(std::ostream &os, const TensorBase &tensor) {
      const auto *cpuStorage =
          dynamic_cast<const NDTensorStorage<T> *>(tensor.get_storage());
      if (cpuStorage) {
         const TensorBuffer &buf = cpuStorage->data();
         const size_t n = cpuStorage->size();
         const T *p = buf.template data_as<const T>();
         os << "Tensor(";
         for (size_t i = 0; i < n; i++) {
            os << p[i]; // NOLINT TODO: change to use view
            if (i + 1 < n) {
               os << ", ";
            }
         }
         os << ")" << std::endl;
      } else {
         os << "Tensor(unsupported storage type)";
      }
      return os;
   }

   std::string shape_str() const {
      std::ostringstream oss;
      oss << '(';
      for (size_t i = 0; i < shape_.size(); ++i) {
         oss << shape_[i];
         if (i + 1 < shape_.size()) {
            oss << ',';
         }
      }
      oss << ')';
      return oss.str();
   }

 protected:
   std::shared_ptr<ITensorStorage<T>> storage_;
   std::vector<std::size_t> shape_{}, strides_{};
   DType dtype_;
   Device device_;
   IAllocator *allocator_ = nullptr;

   void replace_from(TensorBase &&tmp) {
      storage_.swap(tmp.storage());
      shape_.swap(tmp.shape_);
      strides_.swap(tmp.strides_);
   }

   std::shared_ptr<ITensorStorage<T>>
   make_storage(const std::vector<size_t> &shape, std::size_t count,
                Device device, IAllocator *alloc) {
      if (device.is_cpu()) {
         return std::make_shared<NDTensorStorage<T>>(shape, count, device,
                                                     alloc);
      } else if (device.is_gpu() || device.is_cuda()) {
         throw std::runtime_error("GPU is not supported yes");
      } else {
         throw std::runtime_error("Unsupported device");
      }
   }

   std::shared_ptr<ITensorStorage<T>>
   make_storage_with_data(const std::vector<size_t> &shape,
                          std::vector<T> &data, Device device,
                          IAllocator *alloc) {
      if (device.is_cpu()) {
         return std::make_shared<NDTensorStorage<T>>(shape, std::move(data),
                                                     device, alloc);
      } else if (device.is_gpu() || device.is_cuda()) {
         throw std::runtime_error("GPU is not supported yes");
      } else {
         throw std::runtime_error("Unsupported device");
      }
   }
};

#endif // TENSOR_BASE_HPP
