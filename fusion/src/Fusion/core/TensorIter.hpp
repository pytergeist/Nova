// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef TENSOR_ITER_HPP
#define TENSOR_ITER_HPP

#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "Fusion/common/Checks.hpp"
#include "Fusion/cpu/SimdTraits.hpp"
#include "Fusion/cpu/simd/VecNeon128.hpp"

#include "PlanMeta.hpp"
#include "TensorPlan.h"

namespace fusion {

namespace iter {

template <typename IterPlan, std::size_t N, class InnerFn>
inline void walk(int dim, const int inn, const IterPlan &plan,
                 std::array<uint8_t *, N> &ptr, InnerFn &&inner) {
   if (dim == inn) {
      const auto &ld = plan.loop[inn];
      std::vector<int64_t> s = ld.stride_bytes;
      inner(ptr, ld.size, s);
      return;
   }
   const auto &ld = plan.loop[dim];
   for (int64_t i = 0; i < ld.size; ++i) {
      walk(dim + 1, inn, plan, ptr, inner);
      for (int k = 0; k < plan.num_operands; ++k)
         ptr[k] += ld.stride_bytes[k];
   }
   for (int k = 0; k < plan.num_operands; ++k)
      ptr[k] -= ld.stride_bytes[k] * ld.size;
};

template <typename IterPlan, std::size_t N, typename FnInnermost>
inline void for_each_outer_then_inner(const IterPlan &plan,
                                      std::array<uint8_t *, N> &base,
                                      FnInnermost &&inner) {
   // first set the ndim (2 usually, for 2 tensors in loop)
   // set base ptrs (size=3 here, a, b, out?)
   const int ndim = static_cast<int>(plan.loop.size());

   if (ndim == 0) {
      // if ndim = 0 set s vector to num_operands(0)
      // e.g. if num_operands = 3, s = {0, 0, 0}, we then pass into
      // the inner func
      static thread_local std::vector<std::int64_t> zeros;
      zeros.assign(plan.num_operands, 0);
      inner(base, 1, zeros);
      return;
   }

   const int inn = ndim - 1;
   walk(0, inn, plan, base, inner);
}

template <typename T, class Tag>
inline void tag_strided_binary(T *o, const T *a, const T *b, const int64_t &so,
                               const int64_t &sa, const int64_t &sb,
                               const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] = tag(a[i * sa], b[i * sb]);
}

template <typename T, class Tag>
inline void tag_strided_unary(T *o, const T *a, const int64_t &so,
                              const int64_t &sa, const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] = tag(a[i * sa]);
}

template <typename T, class Tag>
inline void tag_fallback_reduction(T *o, const T *a, const int64_t &so,
                                   const int64_t &sa, const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] += tag(a[i * sa]);
}

template <typename T, class Tag, class TensorT>
void binary_ewise_tag(const TensorT &A, const TensorT &B,
                      const BinaryEwiseMeta &meta, TensorT &out) {

   FUSION_CHECK(A.is_initialised(), "binary ewise: LHS uninitialised");
   FUSION_CHECK(B.is_initialised(), "binary ewise: RHS uninitialised");
   FUSION_CHECK(A.is_initialised() && B.is_initialised(),
                "uninitialised tensor");
   std::array<uint8_t *, 3> base = {
       reinterpret_cast<uint8_t *>(const_cast<T *>(out.get_ptr())),
       reinterpret_cast<uint8_t *>(const_cast<T *>(A.get_ptr())),
       reinterpret_cast<uint8_t *>(const_cast<T *>(B.get_ptr()))};

   if (meta.fastpath) {
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const auto *b = reinterpret_cast<const T *>(base[2]);
      const size_t len = meta.fast_len;
      if constexpr (simd_traits<Tag, T>::available) {
         simd_traits<Tag, T>::execute_contiguous(a, b, o, len, false, false);
      } else {
         tag_strided_binary<T, Tag>(o, a, b, 1, 1, 1, len);
      }
      return;
   }

   for_each_outer_then_inner<BroadcastPlan, 3>(
       meta.plan, base,
       [&](std::array<uint8_t *, 3> &p, int64_t len,
           const std::vector<int64_t> &sbytes) {
          const auto step = static_cast<int64_t>(
              sizeof(T)); // sizeof(T) = size of datatype (n bytes)
          const bool out_contig =
              (sbytes[0] == step); // true if s[0] == step (e.g. bytes)
          // below here means that a/b must be either 0 (for broadcast) or same
          // size as step (e.g. bytes)
          const bool a_unit =
              (sbytes[1] == 0 ||
               sbytes[1] == step); // a_unit = True if s[1] = 0 or s[1] = bytes
          const bool b_unit =
              (sbytes[2] == 0 ||
               sbytes[2] == step); // b_unit = True if s[1] = 0 or s[1] = bytes

          auto *o = reinterpret_cast<T *>(
              p[0]); // takes bytes ptr and treats it as if it were a ptr to T
          // (dtype from template)
          const auto *a = reinterpret_cast<const T *>(p[1]); // same as above
          const auto *b = reinterpret_cast<const T *>(p[2]); // same as above

          if constexpr (simd_traits<Tag, T>::available) {
             // include simd impl
             // from traits availible
             if (out_contig && a_unit && b_unit && len > 0) {
                // if all true continue
                const bool a_scalar =
                    (sbytes[1] ==
                     0); // true (a is scalar) if s[1] == 0 (broadcast)
                const bool b_scalar =
                    (sbytes[2] ==
                     0); // true (b is scalar) if s[1] == 0 (broadcast)
                // if all above true execute the contigous op from simd_traits
                simd_traits<Tag, T>::execute_contiguous(
                    a, b, o, static_cast<size_t>(len), a_scalar, b_scalar);
                return;
             }
             const bool a_unit = (sbytes[1] == step);
             const bool b_unit = (sbytes[2] == step);
             if (out_contig && (a_unit || b_unit)) {
                const int64_t so = 1;
                const int64_t sa = sbytes[1] / step;
                const int64_t sb = sbytes[2] / step;
                tag_strided_binary<T, Tag>(o, a, b, so, sa, sb, len);
                return;
             }
          }

          const int64_t so = sbytes[0] / step;
          const int64_t sa = (sbytes[1] == 0) ? 0 : sbytes[1] / step;
          const int64_t sb = (sbytes[2] == 0) ? 0 : sbytes[2] / step;
          // uses tag struct from simd tags as fallback
          tag_strided_binary<T, Tag>(o, a, b, so, sa, sb, len);
       });
}

template <typename T, class Tag, class TensorT>
void unary_ewise_tag(const TensorT &A, UnaryEwiseMeta &meta,
                     TensorT &out_data) {

   std::array<uint8_t *, 2> base = {
       reinterpret_cast<uint8_t *>(out_data.get_ptr()),
       reinterpret_cast<uint8_t *>(const_cast<T *>(A.get_ptr())),
   };

   if (meta.fastpath) { // TODO: is contig check correct here?
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const size_t len = meta.fast_len;
      if constexpr (simd_traits<Tag, T>::available) {
         simd_traits<Tag, T>::execute_contiguous(a, o, len, false);
      } else {
         tag_strided_unary<T, Tag>(o, a, 1, 1, len);
      }
      return;
   }

   for_each_outer_then_inner<BroadcastPlan, 2>(
       meta.plan, base,
       [&](std::array<uint8_t *, 2> &p, int64_t len,
           const std::vector<int64_t> &sbytes) {
          const auto step = static_cast<int64_t>(
              sizeof(T)); // sizeof(T) = size of datatype (n bytes)
          const bool out_contig =
              (sbytes[0] == step); // true if s[0] == step (e.g. bytes)
          // below here means that a/b must be either 0 (for broadcast) or same
          // size as step (e.g. bytes)
          const bool a_unit =
              (sbytes[1] == 0 ||
               sbytes[1] ==
                   step); // a_unit = True if s[1] = 0 or s[1] = bytes (4_

          auto *o = reinterpret_cast<T *>(
              p[0]); // takes bytes ptr and treats it as if it were a ptr to T
          // (dtype from template)
          const auto *a = reinterpret_cast<const T *>(p[1]); // same as above

          if constexpr (simd_traits<Tag, T>::available) {
             // include simd impl
             // from traits availible
             if (out_contig && a_unit && len > 0) {
                // if all true continue
                const bool a_scalar =
                    (sbytes[1] ==
                     0); // true (a is scalar) if s[1] == 0 (broadcast)
                // if all above true execute the contigous op from simd_traits
                simd_traits<Tag, T>::execute_contiguous(
                    a, o, static_cast<size_t>(len), a_scalar);
                return;
             }
             const bool a_unit = (sbytes[1] == step);
             if (out_contig && a_unit) {
                const int64_t so = 1;
                const int64_t sa = sbytes[1] / step;
                tag_strided_unary<T, Tag>(o, a, so, sa, len);
                return;
             }
          }

          const int64_t so = sbytes[0] / step;
          const int64_t sa = (sbytes[1] == 0) ? 0 : sbytes[1] / step;
          // uses tag struct from simd tags as fallback
          Tag tag{};
          tag_strided_unary<T, Tag>(o, a, so, sa, len);
       });
}

template <typename T, class Tag, class TensorT>
void reduction_tag(const TensorT &A, ReductionMeta &meta, TensorT &out_data) {

   auto *out = reinterpret_cast<T *>(out_data.get_ptr());
   std::fill(out, out + out_data.flat_size(), T{0});
   std::array<uint8_t *, 2> base = {
       reinterpret_cast<uint8_t *>(const_cast<T *>(out)),
       reinterpret_cast<uint8_t *>(const_cast<T *>(A.get_ptr())),
   };

   if (meta.fastpath) {
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const size_t len = meta.fast_len;
      if constexpr (simd_traits<Tag, T>::available) {
         *o += simd_traits<Tag, T>::reduce_contiguous(a, len);
      } else {
         tag_fallback_reduction<T, Tag>(o, a, 1, 1, len);
      }
      return;
   }

   for_each_outer_then_inner<ReductionPlan, 2>(
       meta.plan, base,
       [&](std::array<uint8_t *, 2> &p, int64_t len,
           const std::vector<int64_t> &sbytes) {
          const auto step = static_cast<int64_t>(sizeof(T));
          const bool out_contig = (sbytes[0] == 0);
          const bool a_ok = (sbytes[1] == 0 || sbytes[1] == step);

          auto *o = reinterpret_cast<T *>(p[0]);
          const auto *a = reinterpret_cast<const T *>(p[1]);

          if constexpr (simd_traits<Tag, T>::available) {
             if (sbytes[0] == 0 && sbytes[1] == step && len > 0) {
                *o += simd_traits<Tag, T>::reduce_contiguous(
                    a, static_cast<size_t>(len));
                return;
             }
          }

          const std::int64_t so = sbytes[0] / step;
          const std::int64_t sa = (sbytes[1] == 0) ? 0 : sbytes[1] / step;
          Tag tag{};
          tag_fallback_reduction<T, Tag>(o, a, so, sa, len);
       });
}

} // namespace iter

} // namespace fusion

#endif // TENSOR_ITER_HPP
