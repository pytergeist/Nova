#ifndef FUSION_CORE_TENSOR_HPP
#define FUSION_CORE_TENSOR_HPP

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "AoSoATensor.hpp"
#include "DenseTensor.hpp"
#include "Fusion/core/tensor/SoATensor.hpp"

#include "Fusion/common/Checks.hpp"
#include "Fusion/core/fuir/OperandDescription.h"

template <typename T> class Tensor {
 public:
   using value_type = T;

   using Dense = DenseTensor<T>;
   using SoA = SoATensor<T>;
   using AoSoA = AoSoATensor<T>;

   static constexpr std::string_view name = "Tensor";

   Tensor() = default;

   explicit Tensor(Dense dense)
       : storage_(std::move(dense)), layout_(fusion::core::LayoutKind::Dense) {}

   explicit Tensor(SoA soa)
       : storage_(std::move(soa)), layout_(fusion::core::LayoutKind::SoA) {}

   explicit Tensor(AoSoA blocked)
       : storage_(std::move(blocked)),
         layout_(fusion::core::LayoutKind::AoSoA) {}

   static Tensor from_dense(Dense dense) { return Tensor(std::move(dense)); }

   static Tensor from_soa(SoA soa) { return Tensor(std::move(soa)); }

   static Tensor from_aosoa(AoSoA blocked) {
      return Tensor(std::move(blocked));
   }

   fusion::core::LayoutKind layout() const noexcept { return layout_; }

   bool is_dense() const noexcept {
      return layout_ == fusion::core::LayoutKind::Dense ||
             layout_ == fusion::core::LayoutKind::Strided;
   }

   bool is_soa() const noexcept {
      return layout_ == fusion::core::LayoutKind::SoA;
   }

   bool is_aosoa() const noexcept {
      return layout_ == fusion::core::LayoutKind::AoSoA;
   }

   bool is_initialised() const {
      return std::visit([](const auto &x) { return x.base().is_initialised(); },
                        storage_);
   }

   bool empty() const {
      return std::visit([](const auto &x) { return x.base().empty(); },
                        storage_);
   }

   DType dtype() const { return physical_base().dtype(); }

   Device device() const { return physical_base().device(); }

   std::vector<std::size_t> shape() const {
      if (is_dense()) {
         return std::get<Dense>(storage_).shape();
      }
      throw std::runtime_error("Currently only Dense shape supported");
   }

   std::size_t rank() const { return logical_shape().size(); }

   std::size_t ndims() const { return rank(); }

   std::size_t size() const { return physical_base().size(); }

   std::size_t flat_size() const { return physical_base().flat_size(); }

   void clear() noexcept { physical_base().clear(); }

   T *begin() { return physical_base().begin(); }

   T *end() { return physical_base().end(); }

   const T *begin() const { return physical_base().begin(); }

   const T *end() const { return physical_base().end(); }

   TensorBuffer &raw_data() { return physical_base().raw_data(); }

   const TensorBuffer &raw_data() const { return physical_base().raw_data(); }
   std::vector<std::size_t> physical_shape() const {
      return physical_base().shape();
   }

   std::vector<std::int64_t> physical_strides() const {
      return physical_base().strides();
   }

   std::vector<std::int64_t> strides() const {
      return physical_base().strides();
   }

   std::vector<std::size_t> logical_shape() const {
      return std::visit([](const auto &x) { return x.logical_shape(); },
                        storage_);
   }

   std::vector<std::size_t> storage_shape() const {
      return std::visit([](const auto &x) { return x.storage_shape(); },
                        storage_);
   }

   std::size_t physical_rank() const { return physical_base().rank(); }

   std::size_t physical_ndims() const { return physical_base().ndims(); }

   std::size_t physical_size() const { return physical_base().size(); }

   Dense &dense() {
      FUSION_CHECK(is_dense(), "Tensor: not Dense layout");
      return std::get<Dense>(storage_);
   }

   const Dense &dense() const {
      FUSION_CHECK(is_dense(), "Tensor: not Dense layout");
      return std::get<Dense>(storage_);
   }

   SoA &soa() {
      FUSION_CHECK(is_soa(), "Tensor: not SoA layout");
      return std::get<SoA>(storage_);
   }

   const SoA &soa() const {
      FUSION_CHECK(is_soa(), "Tensor: not SoA layout");
      return std::get<SoA>(storage_);
   }

   AoSoA &aosoa() {
      FUSION_CHECK(is_aosoa(), "Tensor: not AoSoA layout");
      return std::get<AoSoA>(storage_);
   }

   const AoSoA &aosoa() const {
      FUSION_CHECK(is_aosoa(), "Tensor: not AoSoA layout");
      return std::get<AoSoA>(storage_);
   }

   DenseTensor<T> &physical_base() {
      return std::visit(
          []<typename T0>(T0 &x) -> DenseTensor<T> & {
             if constexpr (std::is_same_v<std::decay_t<T0>, DenseTensor<T>>) {
                return x;
             } else {
                return x.base();
             }
          },
          storage_);
   }

   const DenseTensor<T> &physical_base() const {
      return std::visit(
          []<typename T0>(const T0 &x) -> const DenseTensor<T> & {
             if constexpr (std::is_same_v<std::decay_t<T0>, DenseTensor<T>>) {
                return x;
             } else {
                return x.base();
             }
          },
          storage_);
   }

   T *data_ptr() { return physical_base().get_ptr(); }

   const T *data_ptr() const { return physical_base().get_ptr(); }

   Tensor operator+(const Tensor &other) const {
      require_dense_binary(other, "add");
      return Tensor::from_dense(dense() + other.dense());
   }

   Tensor operator-(const Tensor &other) const {
      require_dense_binary(other, "sub");
      return Tensor::from_dense(dense() - other.dense());
   }

   Tensor operator*(const Tensor &other) const {
      require_dense_binary(other, "mul");
      return Tensor::from_dense(dense() * other.dense());
   }

   Tensor operator/(const Tensor &other) const {
      require_dense_binary(other, "div");
      return Tensor::from_dense(dense() / other.dense());
   }

   Tensor operator+(T scalar) const {
      require_dense("scalar add");
      return Tensor::from_dense(dense() + scalar);
   }

   Tensor operator-(T scalar) const {
      require_dense("scalar sub");
      return Tensor::from_dense(dense() - scalar);
   }

   Tensor operator*(T scalar) const {
      require_dense("scalar mul");
      return Tensor::from_dense(dense() * scalar);
   }

   Tensor operator/(T scalar) const {
      require_dense("scalar div");
      return Tensor::from_dense(dense() / scalar);
   }

   Tensor matmul(const Tensor &other) const {
      require_dense_binary(other, "matmul");
      return Tensor::from_dense(dense().matmul(other.dense()));
   }

   Tensor sum(std::size_t axis, bool keepdim) const {
      require_dense("sum");
      return Tensor::from_dense(dense().sum(axis, keepdim));
   }

   Tensor mean(std::size_t axis, bool keepdim) const {
      require_dense("mean");
      return Tensor::from_dense(dense().mean(axis, keepdim));
   }

   Tensor sqrt() const {
      require_dense("sqrt");
      return Tensor::from_dense(dense().sqrt());
   }

   Tensor log() const {
      require_dense("log");
      return Tensor::from_dense(dense().log());
   }

   Tensor exp() const {
      require_dense("exp");
      return Tensor::from_dense(dense().exp());
   }

   Tensor operator>(const Tensor &other) const {
      require_dense_binary(other, "greater");
      return Tensor::from_dense(dense() > other.dense());
   }

   Tensor operator>=(const Tensor &other) const {
      require_dense_binary(other, "greater_equal");
      return Tensor::from_dense(dense() >= other.dense());
   }

   Tensor operator>=(T scalar) const {
      require_dense("scalar greater_equal");
      return Tensor::from_dense(dense() >= scalar);
   }

   Tensor maximum(const Tensor &other) const {
      require_dense_binary(other, "maximum");
      return Tensor::from_dense(dense().maximum(other.dense()));
   }

   Tensor maximum(T scalar) const {
      require_dense("scalar maximum");
      return Tensor::from_dense(dense().maximum(scalar));
   }

   Tensor pow(const Tensor &other) const {
      require_dense_binary(other, "pow");
      return Tensor::from_dense(dense().pow(other.dense()));
   }

   Tensor pow(T scalar) const {
      require_dense("scalar pow");
      return Tensor::from_dense(dense().pow(scalar));
   }

   Tensor reciprocal() const {
      require_dense("reciprocal");
      return Tensor::from_dense(dense().reciprocal());
   }

   Tensor swapaxes(int axis1, int axis2) const {
      require_dense("swapaxes");
      return Tensor::from_dense(dense().swapaxes(axis1, axis2));
   }

   friend std::ostream &operator<<(std::ostream &os, const Tensor &t) {
      return os << t.physical_base();
   }

 private:
   using Storage = std::variant<Dense, SoA, AoSoA>;

   Storage storage_{Dense{}};
   fusion::core::LayoutKind layout_{fusion::core::LayoutKind::Dense};

   void require_dense(const char *op_name) const {
      FUSION_CHECK(is_dense(), std::string("Tensor::") + op_name +
                                   " currently requires Dense layout");
   }

   void require_dense_binary(const Tensor &other, const char *op_name) const {
      FUSION_CHECK(is_dense() && other.is_dense(),
                   std::string("Tensor::") + op_name +
                       " currently requires Dense layouts");
   }
};

template <typename T>
Tensor<T> tensor_scalar_t(T scalar, DType dtype = DType::FLOAT32,
                          Device device = Device{DeviceType::CPU, 0}) {
   return Tensor<T>::from_dense(scalar_t(scalar, dtype, device));
}

#endif // FUSION_CORE_TENSOR_HPP