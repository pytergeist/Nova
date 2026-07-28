#ifndef FUSION_CORE_TENSOR_DENSE_TENSOR
#define FUSION_CORE_TENSOR_DENSE_TENSOR

#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../dtype/Dtype.h"
#include "Fusion/core/memory/alloc/DefaultAllocator.h"
#include "Fusion/common/Checks.hpp"
#include "Fusion/core/Layout.h"
#include "Fusion/core/device/Device.h"
#include "Fusion/ops/contraction/GeMM.hpp"
#include "Fusion/ops/elementwise/Arithmetic.hpp"
#include "Fusion/ops/elementwise/Comparison.hpp"
#include "Fusion/ops/elementwise/Unary.hpp"
#include "Fusion/ops/reduction/Reduce.hpp"
#include "Fusion/core/memory/storage/DenseStorage.hpp"
#include "Fusion/core/memory/storage/StorageInterface.hpp"
#include "TensorView.hpp"

#include "Fusion/common/Log.hpp"

template <typename T> // TODO: need to either pass in device somehow?
DenseTensor<T> scalar_t(const T scalar, const DType dtype = DType::FLOAT32,
                        Device device = Device{DeviceType::CPU, 0}) {
   return DenseTensor<T>{{1}, {scalar}, dtype, device};
}

template <typename T> class DenseTensor {
 public:
   static constexpr std::string_view name = "RawTensor";
   using value_type = T;

   DenseTensor() : storage_(nullptr), device_(Device{DeviceType::CPU, 0}) {}

   DenseTensor(const DenseTensor &) = default;
   DenseTensor &operator=(const DenseTensor &) = default;

   DenseTensor(DenseTensor &&) noexcept = default;
   DenseTensor &operator=(DenseTensor &&) noexcept = default;

   ~DenseTensor() = default;

   // TODO: It is more idiomatic and less bug prone for the context to own
   // tensor alloc, not to have the allocation strategy as a property of the
   // tensor itself e.g. `alloc.make_tensor(...)` and later
   // `arena.make_tensor(...)`, 'slab.make_tensor(...).
   explicit DenseTensor(std::vector<std::size_t> shape, std::vector<T> data,
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

   explicit DenseTensor(std::vector<size_t> shape, DType dtype, Device device,
                        IAllocator *allocator = nullptr)
       : shape_(std::move(shape)), dtype_(dtype), device_(device) {
      FUSION_CHECK(device.is_cpu(), "Unsupported device type");
      FUSION_CHECK(!shape_.empty(), "Tensor: empty shape");
      std::size_t sz = set_contiguous_strides();
      auto *alloc = allocator ? allocator : &default_allocator();
      storage_ = make_storage(shape_, sz, device_, alloc);
   }

   DType dtype() const noexcept { return dtype_; }
   std::size_t dtype_size() const { return get_dtype_size(dtype_); }

   std::size_t rank() const { return shape_.size(); }
   std::size_t ndims() const { return shape_.size(); }
   std::vector<std::size_t> shape() const { return shape_; }
   std::vector<std::size_t> storage_shape() const { return shape_; }
   std::vector<std::size_t> logical_shape() const { return shape_; }
   std::vector<std::int64_t> strides() const { return strides_; }
   Device device() const noexcept { return device_; }

   bool is_contiguous() const noexcept {
      return fusion::core::calc_contiguous(shape_, strides_);
   }

   std::size_t size() const noexcept {
      return storage_->data().template size<T>();
   }

   bool empty() const noexcept { return !storage_ || storage_->data().empty(); }

   bool is_initialised() const noexcept { return storage_ != nullptr; }
   std::size_t flat_size() const { return storage_->size(); }

   ITensorStorage<T> *get_storage() { return storage_.get(); }
   const ITensorStorage<T> *get_storage() const { return storage_.get(); }

   const std::shared_ptr<ITensorStorage<T>> &storage() const noexcept {
      return storage_;
   }
   std::shared_ptr<ITensorStorage<T>> &storage() noexcept { return storage_; }

   std::size_t storage_use_count() const noexcept {
      return storage_.use_count();
   }

   TensorBuffer &raw_data() { return storage_->data(); }
   const TensorBuffer &raw_data() const { return storage_->data(); }

   DenseTensor<T> &base() noexcept { return *this; }
   const DenseTensor<T> &base() const noexcept { return *this; }

   T *get_ptr() { return storage_->data_ptr(); }
   const T *get_ptr() const { return storage_->data_ptr(); }

   TensorView<T>
   view() { // TODO: need to eventually pass into metadata for views
      return TensorView<T>(storage_->data().template data<T>(), this->shape(),
                           this->strides(), this->rank(), this->ndims());
   }

   bool is_view() const noexcept { return is_view_; }
   std::size_t storage_offset() const noexcept { return storage_offset_elems_; }

   T operator[](int idx) const {
      return storage_->data().template data_as<const T>()[idx];
   }

   T *begin() { return storage_->data().template begin<T>(); }
   T *end() { return storage_->data().template end<T>(); }

   T *begin() const { return storage_->data().template begin<T>(); }
   T *end() const { return storage_->data().template end<T>(); }

   std::size_t set_contiguous_strides() {
      std::int64_t sz = 1;
      strides_.assign(shape_.size(), 0);

      for (std::int64_t i = static_cast<std::int64_t>(shape_.size()) - 1;
           i >= 0; --i) {
         strides_[static_cast<std::size_t>(i)] = sz;
         sz *= static_cast<std::int64_t>(shape_[static_cast<std::size_t>(i)]);
      }
      return static_cast<std::size_t>(sz);
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

   void assign(const DenseTensor &other) {
      if (!storage_) {
         *this = other;
      } else {
         FUSION_CHECK(shape_ == other.shape(), "assign shape mismatch");
         FUSION_CHECK(dtype_ == other.dtype(), "assign dtype mismatch");
         FUSION_CHECK(device_ == other.device(), "assign device mismatch");
         storage_->data().assign(other.begin(), other.end());
      }
   };

   DenseTensor operator+(const T scalar) const {
      return fusion::ops::aligned::add(*this,
                                       scalar_t(scalar, dtype(), device()));
   }

   DenseTensor operator-(const T scalar) const {
      return fusion::ops::aligned::sub(*this,
                                       scalar_t(scalar, dtype(), device()));
   }

   DenseTensor operator*(const T scalar) const {
      return fusion::ops::aligned::mul(*this,
                                       scalar_t(scalar, dtype(), device()));
   }

   DenseTensor operator/(const T scalar) const {
      return fusion::ops::aligned::div(*this,
                                       scalar_t(scalar, dtype(), device()));
   }

   DenseTensor operator>=(const T scalar) const {
      return fusion::ops::aligned::greater(*this,
                                           scalar_t(scalar, dtype(), device()));
   }

   DenseTensor maximum(const T scalar) const {
      return fusion::ops::aligned::maximum(*this,
                                           scalar_t(scalar, dtype(), device()));
   }

   DenseTensor pow(const T scalar) const {
      return fusion::ops::aligned::pow(*this,
                                       scalar_t(scalar, dtype(), device()));
   }

   DenseTensor operator+(const DenseTensor &other) const {
      return fusion::ops::aligned::add(*this, other);
   }

   DenseTensor operator-(const DenseTensor &other) const {
      return fusion::ops::aligned::sub(*this, other);
   }

   DenseTensor operator*(const DenseTensor &other) const {
      return fusion::ops::aligned::mul(*this, other);
   }

   DenseTensor operator/(const DenseTensor &other) const {
      return fusion::ops::aligned::div(*this, other);
   }

   DenseTensor operator>(const DenseTensor &other) const {
      return fusion::ops::aligned::greater(*this, other);
   }

   DenseTensor operator>=(const DenseTensor &other) const {
      return fusion::ops::aligned::greater(*this, other);
   }

   DenseTensor matmul(const DenseTensor &other) const {
      return fusion::ops::contraction::matmul(*this, other);
   }

   DenseTensor maximum(const DenseTensor &other) const {
      return fusion::ops::aligned::maximum(*this, other);
   }

   DenseTensor pow(const DenseTensor &other) const {
      return fusion::ops::aligned::pow(*this, other);
   }

   DenseTensor reciprocal() const {
      return fusion::ops::aligned::reciprocal(*this);
   }

   DenseTensor sqrt() const { return fusion::ops::aligned::sqrt(*this); }
   DenseTensor log() const { return fusion::ops::aligned::log(*this); }
   DenseTensor exp() const { return fusion::ops::aligned::exp(*this); }

   DenseTensor sum(const std::size_t axis, const bool keepdim) const {
      return fusion::ops::reduction::sum(*this, axis, keepdim);
   }
   DenseTensor mean(const std::size_t axis, const bool keepdim) const {
      return fusion::ops::reduction::mean(*this, axis, keepdim);
   }

   DenseTensor swapaxes(const int axis1, const int axis2) const {
      return fusion::ops::contraction::swapaxes(*this, axis1, axis2);
   }

   DenseTensor &operator-=(const DenseTensor &other) {
      fusion::ops::aligned::sub_inplace(*this, other);
      return *this;
   }

   friend std::ostream &operator<<(std::ostream &os,
                                   const DenseTensor &tensor) {
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
   std::vector<std::size_t> shape_{};
   std::vector<std::int64_t> strides_{};
   bool is_view_{false};
   std::size_t storage_offset_elems_{0};
   DType dtype_;
   Device device_;
   IAllocator *allocator_ = nullptr;

   void replace_from(DenseTensor &&tmp) {
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
      }
      if (device.is_gpu() || device.is_cuda()) {
         throw std::runtime_error("GPU is not supported yes");
      }
      throw std::runtime_error("Unsupported device");
   }
};

#endif // FUSION_CORE_TENSOR_DENSE_TENSOR
