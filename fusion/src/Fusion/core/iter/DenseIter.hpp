#ifndef FUSION_CORE_TENSOR_ITER_HPP
#define FUSION_CORE_TENSOR_ITER_HPP

#include <vector>

#include "Fusion/common/Checks.hpp"
#include "Fusion/cpu/blas/BlasTraits.hpp"
#include "Fusion/cpu/simd/SimdTraits.hpp"

#include "DenseIterPlanView.hpp"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/core/planning/PlanBuilders.h"

namespace fusion::dense::iter {

struct OperandStep {
   fuir::AccessKind kind{fuir::AccessKind::Affine};
   std::int64_t byte_stride{0};
};

template <std::size_t N> struct DenseSegment {
   std::int64_t len{0};
   std::array<uint8_t *, N> ptrs;
   std::array<OperandStep, N> step;
};

template <std::size_t N>
DenseSegment<N> construct_inner_segment(int inner_dim,
                                        const DenseIterPlanView &view,
                                        std::array<uint8_t *, N> &ptr) {
   FUSION_CHECK(inner_dim >= 0, "inner_dim must be non-negative");

   FUSION_CHECK(inner_dim < static_cast<int>(view.loop.size()),
                "inner_dim out of range");

   DenseSegment<N> seg;
   seg.len = static_cast<std::int64_t>(view.loop[inner_dim].size);
   seg.ptrs = ptr;
   for (std::size_t k = 0; k < N; k++) {
      seg.step[k].kind = view.operands[k].access;
      if (seg.step[k].kind == fuir::AccessKind::Affine) {
         seg.step[k].byte_stride =
             view.operands[k].affine.byte_stride_per_loop[inner_dim];
      } else {
         throw std::runtime_error(
             "Access invalid: currently only affine is unsupported");
      }
   }
   return seg;
}

template <std::size_t N>
DenseSegment<N> construct_scalar_segment(const DenseIterPlanView &view,
                                         std::array<uint8_t *, N> &ptr) {
   FUSION_CHECK(view.loop.empty(),
                "construct_scalar_segment: plan must have zero loop dims");

   FUSION_CHECK(view.operands.size() == N,
                "construct_scalar_segment: op_access smaller than N");

   DenseSegment<N> seg;
   seg.len = 1;
   seg.ptrs = ptr;
   for (std::size_t k = 0; k < N; k++) {
      seg.step[k].kind = view.operands[k].access;
      if (seg.step[k].kind == fuir::AccessKind::Affine) {
         seg.step[k].byte_stride = 0;
      } else {
         throw std::runtime_error(
             "Access invalid: currently only affine is unsupported");
      }
   }
   return seg;
}

template <std::size_t N, class InnerFn>
void walk(int dim, const int inn, const DenseIterPlanView &view,
          std::array<uint8_t *, N> &ptr, InnerFn &&inner) {
   if (dim == inn) {
      DenseSegment<N> seg = construct_inner_segment(inn, view, ptr);
      inner(seg);
      return;
   }

   const fuir::LoopDim &ld = view.loop[dim];
   for (int64_t i = 0; i < ld.size; ++i) {
      walk(dim + 1, inn, view, ptr, inner);
      for (int k = 0; k < view.num_operands; ++k) {
         fuir::OperandAccess access = view.operands[k];
         ptr[k] += access.affine.byte_stride_per_loop[dim];
      }
   }
   for (int k = 0; k < view.num_operands; ++k) {
      fuir::OperandAccess access = view.operands[k];
      ptr[k] -= access.affine.byte_stride_per_loop[dim] * ld.size;
   }
};

template <std::size_t N, typename FnInnermost>
void for_each_outer_then_inner(const DenseIterPlanView &view,
                               std::array<uint8_t *, N> &base,
                               FnInnermost &&inner)

{
   const int ndim = static_cast<int>(view.loop.size());

   if (ndim == 0) {
      // TODO: evaluate this impl - possibly introducing sutble numerical bugs
      DenseSegment<N> seg = construct_scalar_segment(view, base);
      inner(seg);
      return;
   }

   const int inn = ndim - 1;
   walk(0, inn, view, base, inner);
}


} // namespace fusion::dense::iter

#endif // FUSION_CORE_TENSOR_ITER_HPP
