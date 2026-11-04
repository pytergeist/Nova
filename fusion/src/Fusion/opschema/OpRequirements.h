#ifndef FUSION_CORE_OPSCHEMA_OP_REQUIREMENTS_H
#define FUSION_CORE_OPSCHEMA_OP_REQUIREMENTS_H

#include "Fusion/opschema/OpTraits.h"

namespace fusion::opschema {
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
                 "Tag must be an ewise unary op");
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
                 "Tag must be reduction operation");
   static_assert(s.inputs.kind == ArityKind::Fixed && s.inputs.arity == 1,
                 "Reduction op must have exactly 1 input");
   static_assert(s.outputs.kind == ArityKind::Fixed && s.outputs.arity == 1,
                 "Reduction op must have exactly 1 output");
   static_assert(s.mutation == MutationKind::OutOfPlace,
                 "Reduction helper expects a pure op");
}

template <class Tag> consteval void require_contraction_out_of_place() {
   constexpr OpSchema s = OpTraits<Tag>::schema;

   static_assert(s.category == OpCategory::Contraction,
                 "Tag must be an contraction op");
   static_assert(s.inputs.kind == ArityKind::Fixed && s.inputs.arity == 2,
                 "Contract op must have exactly 2 inputs");
   static_assert(s.outputs.kind == ArityKind::Fixed && s.outputs.arity == 1,
                 "Contraction op must have exactly 1 output");
   static_assert(s.mutation == MutationKind::OutOfPlace,
                 "Contraction helper expects a pure op");
}
} // namespace fusion::opschema

#endif // FUSION_CORE_OPSCHEMA_OP_REQUIREMENTS_H