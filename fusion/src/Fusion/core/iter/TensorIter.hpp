#ifndef FUSION_CORE_TENSOR_ITER_HPP
#define FUSION_CORE_TENSOR_ITER_HPP

#include <vector>

#include "Fusion/common/Checks.hpp"
#include "Fusion/cpu/blas/BlasTraits.hpp"
#include "Fusion/cpu/simd/SimdTraits.hpp"

#include "Fusion/core/planning/PlanMeta.hpp"
#include "Fusion/core/planning/TensorPlan.h"

namespace fusion::dense::iter {

struct OperandStep {
   AccessKind kind{AccessKind::Affine};
   std::int64_t byte_stride{0};
};

template <std::size_t N> struct InnerSegment {
   std::int64_t len{0};
   std::array<uint8_t *, N> ptrs;
   std::array<OperandStep, N> step;
};

template <typename IterPlan, std::size_t N>
InnerSegment<N> construct_inner_segment(int inner_dim, const IterPlan &plan,
                                        std::array<uint8_t *, N> &ptr) {

   FUSION_CHECK(inner_dim >= 0, "inner_dim must be non-negative");
   FUSION_CHECK(inner_dim < static_cast<int>(plan.loop.size()),
                "inner_dim out of range");

   InnerSegment<N> seg;
   seg.len = static_cast<std::int64_t>(plan.loop[inner_dim].size);
   seg.ptrs = ptr;
   for (std::size_t k = 0; k < N; k++) {
      seg.step[k].kind = plan.op_access[k].access;
      if (seg.step[k].kind == AccessKind::Affine) {
         seg.step[k].byte_stride =
             plan.op_access[k].affine.byte_stride_per_loop[inner_dim];
      } else {
         throw std::runtime_error(
             "Access invalid: currently only affine is unsupported");
      }
   }
   return seg;
}

template <typename IterPlan, std::size_t N>
InnerSegment<N> construct_scalar_segment(const IterPlan &plan,
                                         std::array<uint8_t *, N> &ptr) {

   FUSION_CHECK(plan.loop.empty(),
                "construct_scalar_segment: plan must have zero loop dims");

   FUSION_CHECK(plan.op_access.size() == N,
                "construct_scalar_segment: op_access smaller than N");

   InnerSegment<N> seg;
   seg.len = 1;
   seg.ptrs = ptr;
   for (std::size_t k = 0; k < N; k++) {
      seg.step[k].kind = plan.op_access[k].access;
      if (seg.step[k].kind == AccessKind::Affine) {
         seg.step[k].byte_stride = 0;
      } else {
         throw std::runtime_error(
             "Access invalid: currently only affine is unsupported");
      }
   }
   return seg;
}

template <typename IterPlan, std::size_t N, class InnerFn>
void walk(int dim, const int inn, const IterPlan &plan,
          std::array<uint8_t *, N> &ptr, InnerFn &&inner) {
   if (dim == inn) {
      InnerSegment<N> seg = construct_inner_segment(inn, plan, ptr);
      inner(seg);
      return;
   }

   const LoopDim &ld = plan.loop[dim];
   for (int64_t i = 0; i < ld.size; ++i) {
      walk(dim + 1, inn, plan, ptr, inner);
      for (int k = 0; k < plan.num_operands; ++k) {
         OperandAccess access = plan.op_access[k];
         ptr[k] += access.affine.byte_stride_per_loop[dim];
      }
   }
   for (int k = 0; k < plan.num_operands; ++k) {
      OperandAccess access = plan.op_access[k];
      ptr[k] -= access.affine.byte_stride_per_loop[dim] * ld.size;
   }
};

template <typename IterPlan, std::size_t N, typename FnInnermost>
void for_each_outer_then_inner(const IterPlan &plan,
                               std::array<uint8_t *, N> &base,
                               FnInnermost &&inner)

{
   const int ndim = static_cast<int>(plan.loop.size());

   if (ndim == 0) {
      // TODO: evaluate this impl - possibly introducing sutble numerical bugs
      InnerSegment<N> seg = construct_scalar_segment(plan, base);
      inner(seg);
      return;
   }

   const int inn = ndim - 1;
   walk(0, inn, plan, base, inner);
}

template <typename T, class Tag>
void tag_fallback_binary(T *o, const T *a, const T *b, const int64_t &so,
                         const int64_t &sa, const int64_t &sb,
                         const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] = tag(a[i * sa], b[i * sb]);
}

template <typename T, class Tag>
void tag_fallback_unary(T *o, const T *a, const int64_t &so, const int64_t &sa,
                        const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] = tag(a[i * sa]);
}

template <typename T, class Tag>
void tag_fallback_reduction(T *o, const T *a, const int64_t &so,
                            const int64_t &sa, const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < len; ++i)
      o[i * so] += tag(a[i * sa]);
}

template <typename T, class Tag>
void tag_fallback_contraction(T *o, const T *a, const T *b, const int64_t &so,
                              const int64_t &sa, const int64_t &sb,
                              const std::size_t len) {
   Tag tag{};
   for (int64_t i = 0; i < static_cast<int64_t>(len); ++i) {
      o[i * so] += tag(a[i * sa], b[i * sb]);
   }
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

   if (meta.exec == BinaryExecKind::FlatContiguous) {
      auto *o = reinterpret_cast<T *>(base[0]);
      const auto *a = reinterpret_cast<const T *>(base[1]);
      const auto *b = reinterpret_cast<const T *>(base[2]);
      const size_t len = meta.fast_len;
      if constexpr (simd_traits<Tag, T>::available) {
         simd_traits<Tag, T>::execute_contiguous(a, b, o, len, false, false);
      } else {
         tag_fallback_binary<T, Tag>(o, a, b, 1, 1, 1, len);
      }
      return;
   }
   for_each_outer_then_inner<BroadcastPlan, 3>(
       meta.plan, base, [&](InnerSegment<3> &segment) {
          const std::int64_t step = sizeof(T);
          std::int64_t const out_bytes = segment.step[0].byte_stride;
          std::int64_t const a_bytes = segment.step[1].byte_stride;
          std::int64_t const b_bytes = segment.step[2].byte_stride;

          const bool out_contig = out_bytes == step;
          const bool a_unit = a_bytes == 0 || a_bytes == step;
          const bool b_unit = b_bytes == 0 || b_bytes == step;

          T *o = reinterpret_cast<T *>(segment.ptrs[0]);
          const T *a = reinterpret_cast<const T *>(segment.ptrs[1]);
          const T *b = reinterpret_cast<const T *>(segment.ptrs[2]);

          if constexpr (simd_traits<Tag, T>::available) {
             if (out_contig && a_unit && b_unit && segment.len > 0) {
                const bool a_scalar = a_bytes == 0;
                const bool b_scalar = b_bytes == 0;
                simd_traits<Tag, T>::execute_contiguous(
                    a, b, o, static_cast<size_t>(segment.len), a_scalar,
                    b_scalar);
                return;
             }
             if (out_contig && (a_unit || b_unit)) {
                const std::int64_t so = 1;
                const std::int64_t sa = a_bytes / step;
                const std::int64_t sb = b_bytes / step;
                tag_fallback_binary<T, Tag>(o, a, b, so, sa, sb, segment.len);
                return;
             }
          }

          const int64_t so = out_bytes / step;
          const int64_t sa = a_bytes == 0 ? 0 : a_bytes / step;
          const int64_t sb = b_bytes == 0 ? 0 : b_bytes / step;
          tag_fallback_binary<T, Tag>(o, a, b, so, sa, sb, segment.len);
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
         tag_fallback_unary<T, Tag>(o, a, 1, 1, len);
      }
      return;
   }

   for_each_outer_then_inner<BroadcastPlan, 2>(
       meta.plan, base, [&](InnerSegment<2> &segment) {
          const std::int64_t step = sizeof(T);
          std::int64_t const out_bytes = segment.step[0].byte_stride;
          std::int64_t const a_bytes = segment.step[1].byte_stride;
          const bool out_contig = out_bytes == step;
          const bool a_unit = a_bytes == 0 || a_bytes == step;

          T *o = reinterpret_cast<T *>(segment.ptrs[0]);
          const T *a = reinterpret_cast<const T *>(segment.ptrs[1]);

          if constexpr (simd_traits<Tag, T>::available) {

             if (out_contig && a_unit && segment.len > 0) {
                const bool a_scalar = a_bytes == 0;
                simd_traits<Tag, T>::execute_contiguous(
                    a, o, static_cast<size_t>(segment.len), a_scalar);
                return;
             }
             const bool a_unit = a_bytes == step;
             if (out_contig && a_unit) {
                const int64_t so = 1;
                const int64_t sa = a_bytes / step;
                tag_fallback_unary<T, Tag>(o, a, so, sa, segment.len);
                return;
             }
          }

          const int64_t so = a_bytes / step;
          const int64_t sa = a_bytes == 0 ? 0 : a_bytes / step;
          Tag tag{};
          tag_fallback_unary<T, Tag>(o, a, so, sa, segment.len);
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
       meta.plan, base, [&](InnerSegment<2> &segment) {
          const std::int64_t step = sizeof(T);
          std::int64_t const out_bytes = segment.step[0].byte_stride;
          std::int64_t const a_bytes = segment.step[1].byte_stride;

          T *o = reinterpret_cast<T *>(segment.ptrs[0]);
          const T *a = reinterpret_cast<const T *>(segment.ptrs[1]);

          if constexpr (simd_traits<Tag, T>::available) {
             if (out_bytes == 0 && a_bytes == step && segment.len > 0) {
                *o += simd_traits<Tag, T>::reduce_contiguous(
                    a, static_cast<size_t>(segment.len));
                return;
             }
          }

          const std::int64_t so = out_bytes / step;
          const std::int64_t sa = (a_bytes == 0) ? 0 : a_bytes / step;
          Tag tag{};
          tag_fallback_reduction<T, Tag>(o, a, so, sa, segment.len);
       });
}

template <typename T, class BlasTag, class ScalarTag, class TensorT>
void contraction_tag(const TensorT &A, const TensorT &B, ContractionMeta &meta,
                     TensorT &out_data) {

   auto *out = reinterpret_cast<T *>(out_data.get_ptr());
   std::fill(out, out + out_data.flat_size(), T{0});

   if constexpr (fusion::blas::blas_traits<BlasTag, T>::available) {
      if (meta.plan.gemm_like) {
         const auto &g = meta.plan.gemm;
         if (fusion::blas::blas_traits<BlasTag, T>::can_execute(g)) {
            const T *baseA = reinterpret_cast<const T *>(A.get_ptr());
            const T *baseB = reinterpret_cast<const T *>(B.get_ptr());
            T *baseC = reinterpret_cast<T *>(out_data.get_ptr());
            fusion::blas::blas_traits<BlasTag, T>::execute(baseA, baseB, baseC,
                                                           g, T(1), T(0));
            return;
         }
      }
   }

   std::array<uint8_t *, 3> base = {
       reinterpret_cast<uint8_t *>(out),
       reinterpret_cast<uint8_t *>(const_cast<T *>(A.get_ptr())),
       reinterpret_cast<uint8_t *>(const_cast<T *>(B.get_ptr())),
   };

   for_each_outer_then_inner<ContractionPlan, 3>(
       meta.plan, base, [&](InnerSegment<3> &segment) {
          const int64_t step = sizeof(T);
          std::int64_t const out_bytes = segment.step[0].byte_stride;
          std::int64_t const a_bytes = segment.step[1].byte_stride;
          std::int64_t const b_bytes = segment.step[2].byte_stride;

          auto *o = reinterpret_cast<T *>(segment.ptrs[0]);
          auto *a = reinterpret_cast<const T *>(segment.ptrs[1]);
          auto *b = reinterpret_cast<const T *>(segment.ptrs[2]);

          const int64_t so = out_bytes == 0 ? 0 : out_bytes / step;
          const int64_t sa = a_bytes == 0 ? 0 : a_bytes / step;
          const int64_t sb = b_bytes == 0 ? 0 : b_bytes / step;

          tag_fallback_contraction<T, ScalarTag>(
              o, a, b, so, sa, sb, static_cast<std::size_t>(segment.len));
       });
}

} // namespace fusion::iter

#endif // FUSION_CORE_TENSOR_ITER_HPP
