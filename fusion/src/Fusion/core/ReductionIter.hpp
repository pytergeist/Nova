#ifndef REDUCTION_ITER_HPP
#define REDUCTION_ITER_HPP

#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

#include "Fusion/common/Checks.hpp"
#include "Fusion/cpu/SimdTraits.hpp"
#include "Fusion/cpu/simd/VecNeon128.hpp"

#include "Broadcast.h"
#include "EwiseMeta.hpp"

namespace reduction {

template <std::size_t N, class InnerFn>
inline void walk(int dim, const int inn, const ReductionPlan &plan,
                 std::array<uint8_t *, N> &ptr, InnerFn &&inner) {
   const std::size_t kNumPtrs = 2;
   if (dim == inn) {
      const auto &ld = plan.loop[inn];
      std::vector<int64_t> s = ld.stride_bytes;
      inner(ptr, ld.size, s);
      return;
   }
   const auto &ld = plan.loop[dim];
   for (int64_t i = 0; i < ld.size; ++i) {
      walk(dim + 1, inn, plan, ptr, inner);
      for (int k = 0; k < kNumPtrs; ++k)
         ptr[k] += ld.stride_bytes[k];
   }
   // Rewind the pointer after loop completion
   for (int k = 0; k < kNumPtrs; ++k)
      ptr[k] -= ld.stride_bytes[k] * ld.size;
};

template <std::size_t N, typename FnInnermost>
inline void for_each_outer_then_inner(const ReductionPlan &plan,
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
      zeros.assign(1, 0);
      inner(base, 1, zeros);
      return;
   }

   const int inn = ndim - 1;
   walk(0, inn, plan, base, inner);
}

template <typename T, class Tag>
inline void tag_fallback_reduction(T *o, const T *a, const int64_t &so,
                                   const int64_t &sa, const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] += tag(a[i * sa]);
}

// Tag = ExponentialSIMD / NaturalLogSIMD / ...
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

   for_each_outer_then_inner<2>(
       meta.plan, base,
       [&](std::array<uint8_t *, 2> &p, int64_t len,
           const std::vector<int64_t> &sbytes) {
          const auto step = static_cast<int64_t>(
              sizeof(T)); // sizeof(T) = size of datatype (n bytes)
          const bool out_contig =
              (sbytes[0] == 0); // true if s[0] == step (e.g. bytes)
          // below here means that a/b must be either 0 (for broadcast) or same
          // size as step (e.g. bytes)
          const bool a_ok =
              (sbytes[1] == 0 ||
               sbytes[1] ==
                   step); // a_ok = True if s[1] = 0 or s[1] = bytes (4_

          auto *o = reinterpret_cast<T *>(
              p[0]); // takes bytes ptr and treats it as if it were a ptr to T
          // (dtype from template)
          const auto *a = reinterpret_cast<const T *>(p[1]); // same as above

          // SIMD only allowed for scalar output (global or last-axis reduction)
          if constexpr (simd_traits<Tag, T>::available) {
             if (sbytes[0] == 0 && sbytes[1] == step && len > 0) {
                // safe: scalar accumulation
                *o += simd_traits<Tag, T>::reduce_contiguous(
                    a, static_cast<size_t>(len));
                return;
             }
          }

          const std::int64_t so = sbytes[0] / step;
          const std::int64_t sa = (sbytes[1] == 0) ? 0 : sbytes[1] / step;
          // uses tag struct from simd tags as fallback
          Tag tag{};
          tag_fallback_reduction<T, Tag>(o, a, so, sa, len);
       });
}
} // namespace reduction

#endif // REDUCTION_ITER
