#ifndef OP_HELPERS_HPP
#define OP_HELPERS_HPP

#include "Fusion/core/planning/PlanMeta.hpp"

#include "Fusion/core/opschema/OpTraits.h"

template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const DenseTensor<T> &y,
                                  const BinaryEwiseMeta &m) {
   FUSION_CHECK(x.dtype() == y.dtype(), "dtypes do not match!");
   FUSION_CHECK(x.device() == y.device(), "devices do not match!");
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const UnaryEwiseMeta &m) {
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const ReductionMeta &m) {
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
DenseTensor<T> init_out_from_meta(const DenseTensor<T> &x,
                                  const DenseTensor<T> &y,
                                  const ContractionMeta &m) {
   FUSION_CHECK(x.dtype() == y.dtype(), "dtypes do not match!");
   FUSION_CHECK(x.device() == y.device(), "devices do not match!");
   return DenseTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <class Tag> consteval void require_ewise_binary_out_of_place() {
   constexpr OpSchema s = OpTraits<Tag>::schema;

   static_assert(s.category == OpCategory::EwiseBinary,
                 "Tag must be an ewise binary op");
   static_assert(s.inputs.kind == ArityKind::Fixed && s.inputs.arity == 2,
                 "Ewise binary op must have exactly 2 inputs");
   static_assert(s.outputs.kind == ArityKind::Fixed && s.outputs.arity == 1,
                 "Ewise binary op must have exactly 1 output");
   static_assert(s.mutation == MutationKind::OutOfPlace,
                 "Ewise binary helper expects a pure op");
}

template <class Tag> consteval void require_ewise_unary_out_of_place() {
   constexpr OpSchema s = OpTraits<Tag>::schema;

   static_assert(s.category == OpCategory::EwiseUnary,
                 "Tag must be an ewise binary op");
   static_assert(s.inputs.kind == ArityKind::Fixed && s.inputs.arity == 1,
                 "Ewise Unary op must have exactly 2 inputs");
   static_assert(s.outputs.kind == ArityKind::Fixed && s.outputs.arity == 1,
                 "Ewise Unary op must have exactly 1 output");
   static_assert(s.mutation == MutationKind::OutOfPlace,
                 "Ewise Unary helper expects a pure op");
}

template <class Tag> consteval void require_reduction_out_of_place() {
   constexpr OpSchema s = OpTraits<Tag>::schema;

   static_assert(s.category == OpCategory::Reduction,
                 "Tag must be an ewise binary op");
   static_assert(s.inputs.kind == ArityKind::Fixed && s.inputs.arity == 1,
                 "Reduction op must have exactly 2 inputs");
   static_assert(s.outputs.kind == ArityKind::Fixed && s.outputs.arity == 1,
                 "Reduction op must have exactly 1 output");
   static_assert(s.mutation == MutationKind::OutOfPlace,
                 "Reduction helper expects a pure op");
}

template <class Tag> consteval void require_contraction_out_of_place() {
   constexpr OpSchema s = OpTraits<Tag>::schema;

   static_assert(s.category == OpCategory::Contraction,
                 "Tag must be an ewise binary op");
   static_assert(s.inputs.kind == ArityKind::Fixed && s.inputs.arity == 2,
                 "Ewise binary op must have exactly 2 inputs");
   static_assert(s.outputs.kind == ArityKind::Fixed && s.outputs.arity == 1,
                 "Ewise binary op must have exactly 1 output");
   static_assert(s.mutation == MutationKind::OutOfPlace,
                 "Ewise binary helper expects a pure op");
}

#endif // OP_HELPERS_HPP
