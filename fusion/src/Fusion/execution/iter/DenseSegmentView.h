#ifndef FUSION_CORE_ITER_DENSE_SEGMENT_VIEW_HPP
#define FUSION_CORE_ITER_DENSE_SEGMENT_VIEW_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "Fusion/common/Checks.hpp"

namespace fusion::dense::iter {

struct [[nodiscard]] ByteStride {
   std::int64_t stride{0};

   [[nodiscard]] constexpr bool is_broadcast() const noexcept {
      return stride == 0;
   }

   template <typename T>
   [[nodiscard]] constexpr bool is_contiguous() const noexcept {
      return std::cmp_equal(stride, sizeof(T));
   }

   template <typename T>
   [[nodiscard]] constexpr bool is_contiguous_or_broadcast() const noexcept {
      return is_broadcast() || is_contiguous<T>();
   }

   template <typename T> [[nodiscard]] std::int64_t element_stride() const {
      constexpr std::int64_t itemsize = static_cast<std::int64_t>(sizeof(T));

      FUSION_CHECK(stride % itemsize == 0,
                   "Byte stride must be divisible by the element size");

      return stride / itemsize;
   }
};

constexpr ByteStride to_byte_stride(const std::int64_t stride) noexcept {
   return ByteStride{.stride = stride};
}

template <std::size_t NumInputs, std::size_t NumOutputs>
struct DenseSegmentView {
   std::array<std::byte *, NumOutputs> outputs{};
   std::array<const std::byte *, NumInputs> inputs{};

   std::array<ByteStride, NumOutputs> output_byte_stride{};
   std::array<ByteStride, NumInputs> input_byte_stride{};

   std::int64_t len{0};

   [[nodiscard]] constexpr bool empty() const noexcept { return len <= 0; }

   template <typename T>
   [[nodiscard]] T *output(std::size_t index) const noexcept {
      return reinterpret_cast<T *>(outputs[index]);
   }

   template <typename T>
   [[nodiscard]] const T *input(std::size_t index) const noexcept {
      return reinterpret_cast<const T *>(inputs[index]);
   }

   constexpr ByteStride output_stride(std::size_t index) const noexcept {
      return output_byte_stride[index];
   }

   constexpr ByteStride input_stride(std::size_t index) const noexcept {
      return input_byte_stride[index];
   }
};

} // namespace fusion::dense::iter

#endif // FUSION_CORE_ITER_DENSE_SEGMENT_VIEW_HPP