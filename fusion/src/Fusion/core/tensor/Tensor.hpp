#ifndef FUSION_CORE_TENSOR_HPP
#define FUSION_CORE_TENSOR_HPP

#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "RawTensor.hpp"
#include "SoATensor.hpp"
#include "AoSoATensor.hpp"

#include "Fusion/common/Checks.hpp"
#include "Fusion/core/fuir/Descs.h"

template <typename T>
class Tensor {
public:
   using value_type = T;

   using Dense = RawTensor<T>;
   using SoA = SoATensor<T>;
   using BlockedSoA = BlockedSoATensor<T>;

   static constexpr std::string_view name = "Tensor";

   Tensor() = default;

   explicit Tensor(Dense dense)
      : storage_(std::move(dense)), layout_(LayoutKind::Dense) {}

   explicit Tensor(SoA soa)
      : storage_(std::move(soa)), layout_(LayoutKind::SoA) {}

   explicit Tensor(BlockedSoA blocked)
      : storage_(std::move(blocked)), layout_(LayoutKind::AoSoA) {}

   static Tensor from_dense(Dense dense) {
      return Tensor(std::move(dense));
   }

   static Tensor from_soa(SoA soa) {
      return Tensor(std::move(soa));
   }

   static Tensor from_blocked_soa(BlockedSoA blocked) {
      return Tensor(std::move(blocked));
   }

   LayoutKind layout() const noexcept {
      return layout_;
   }

   bool is_dense() const noexcept {
      return layout_ == LayoutKind::Dense || layout_ == LayoutKind::Strided;
   }

   bool is_soa() const noexcept {
      return layout_ == LayoutKind::SoA;
   }

   bool is_blocked_soa() const noexcept {
      return layout_ == LayoutKind::AoSoA;
   }

   bool is_initialised() const {
      return std::visit(
         [](const auto& x) { return x.raw().is_initialised(); },
         storage_
      );
   }

   bool empty() const {
      return std::visit(
         [](const auto& x) { return x.raw().empty(); },
         storage_
      );
   }

   DType dtype() const {
      return physical_raw().dtype();
   }

   Device device() const {
      return physical_raw().device();
   }

   std::vector<std::size_t> physical_shape() const {
      return physical_raw().shape();
   }

   std::vector<std::int64_t> physical_strides() const {
      return physical_raw().strides();
   }

   std::vector<std::size_t> logical_shape() const {
      return std::visit(
         [](const auto& x) { return x.logical_shape(); },
         storage_
      );
   }

   std::vector<std::size_t> storage_shape() const {
      return std::visit(
         [](const auto& x) { return x.storage_shape(); },
         storage_
      );
   }

   std::size_t physical_rank() const {
      return physical_raw().rank();
   }

   std::size_t physical_ndims() const {
      return physical_raw().ndims();
   }

   std::size_t physical_size() const {
      return physical_raw().size();
   }

   Dense& dense() {
      FUSION_CHECK(is_dense(), "Tensor: not Dense layout");
      return std::get<Dense>(storage_);
   }

   const Dense& dense() const {
      FUSION_CHECK(is_dense(), "Tensor: not Dense layout");
      return std::get<Dense>(storage_);
   }

   SoA& soa() {
      FUSION_CHECK(is_soa(), "Tensor: not SoA layout");
      return std::get<SoA>(storage_);
   }

   const SoA& soa() const {
      FUSION_CHECK(is_soa(), "Tensor: not SoA layout");
      return std::get<SoA>(storage_);
   }

   BlockedSoA& blocked_soa() {
      FUSION_CHECK(is_blocked_soa(), "Tensor: not BlockedSoA layout");
      return std::get<BlockedSoA>(storage_);
   }

   const BlockedSoA& blocked_soa() const {
      FUSION_CHECK(is_blocked_soa(), "Tensor: not BlockedSoA layout");
      return std::get<BlockedSoA>(storage_);
   }

   RawTensor<T>& physical_raw() {
      return std::visit(
         [](auto& x) -> RawTensor<T>& {
            if constexpr (std::is_same_v<std::decay_t<decltype(x)>, RawTensor<T>>) {
               return x;
            } else {
               return x.raw();
            }
         },
         storage_
      );
   }

   const RawTensor<T>& physical_raw() const {
      return std::visit(
         [](const auto& x) -> const RawTensor<T>& {
            if constexpr (std::is_same_v<std::decay_t<decltype(x)>, RawTensor<T>>) {
               return x;
            } else {
               return x.raw();
            }
         },
         storage_
      );
   }

   T* data_ptr() {
      return physical_raw().get_ptr();
   }

   const T* data_ptr() const {
      return physical_raw().get_ptr();
   }

   OperandDescription desc(UpdateKind update = UpdateKind::ReadOnly) const {
      switch (layout_) {
      case LayoutKind::Dense:
      case LayoutKind::Strided:
         return dense_desc(update);
      case LayoutKind::SoA:
         return soa_desc(update);
      case LayoutKind::AoSoA:
         return blocked_soa_desc(update);
      default:
         throw std::runtime_error("Tensor::desc: unsupported layout");
      }
   }

   Tensor operator+(const Tensor& other) const {
      require_dense_binary(other, "add");
      return Tensor::from_dense(dense() + other.dense());
   }

   Tensor operator-(const Tensor& other) const {
      require_dense_binary(other, "sub");
      return Tensor::from_dense(dense() - other.dense());
   }

   Tensor operator*(const Tensor& other) const {
      require_dense_binary(other, "mul");
      return Tensor::from_dense(dense() * other.dense());
   }

   Tensor operator/(const Tensor& other) const {
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

   Tensor matmul(const Tensor& other) const {
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

   friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
      return os << t.physical_raw();
   }

private:
   using Storage = std::variant<Dense, SoA, BlockedSoA>;

   Storage storage_{Dense{}};
   LayoutKind layout_{LayoutKind::Dense};

   void require_dense(const char* op_name) const {
      FUSION_CHECK(
         is_dense(),
         std::string("Tensor::") + op_name + " currently requires Dense layout"
      );
   }

   void require_dense_binary(const Tensor& other, const char* op_name) const {
      FUSION_CHECK(
         is_dense() && other.is_dense(),
         std::string("Tensor::") + op_name + " currently requires Dense layouts"
      );
   }

   OperandDescription dense_desc(UpdateKind update) const {
      const auto& x = dense();

      return OperandDescription{
         .shape = x.shape(),
         .strides = x.strides(),
         .itemsize = x.dtype_size(),
         .layout = x.is_contiguous() ? LayoutKind::Dense : LayoutKind::Strided,
         .access = AccessKind::Affine,
         .storage = x.is_view() ? StorageKind::View : StorageKind::Owned,
         .update = update,
         .type = OperandDescType::Tensor,
      };
   }

   OperandDescription soa_desc(UpdateKind update) const {
      const auto& x = soa();
      const auto& raw = x.raw();

      return OperandDescription{
         .shape = x.storage_shape(),
         .strides = raw.strides(),
         .itemsize = raw.dtype_size(),
         .layout = LayoutKind::SoA,
         .access = AccessKind::Affine,
         .storage = StorageKind::Owned,
         .update = update,
         .type = OperandDescType::Tensor,
      };
   }

   OperandDescription blocked_soa_desc(UpdateKind update) const {
      const auto& x = blocked_soa();
      const auto& raw = x.raw();

      return OperandDescription{
         .shape = x.storage_shape(),
         .strides = raw.strides(),
         .itemsize = raw.dtype_size(),
         .layout = LayoutKind::AoSoA,
         .access = AccessKind::Blocked,
         .storage = StorageKind::Owned,
         .update = update,
         .type = OperandDescType::Tensor,
      };
   }
};

template <typename T>
inline Tensor<T> tensor_scalar_t(
   T scalar,
   DType dtype = DType::FLOAT32,
   Device device = Device{DeviceType::CPU, 0}
) {
   return Tensor<T>::from_dense(scalar_t(scalar, dtype, device));
}

#endif // FUSION_CORE_TENSOR_HPP