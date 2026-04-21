#ifndef FUSION_CORE_OP_SCHEMA_OP_TRAITS_H
#define FUSION_CORE_OP_SCHEMA_OP_TRAITS_H

#include <string_view>

#include "OpSchema.h"
#include "OpTags.h"

/// This file contains OpTrait definitions, that

template <class Op> struct OpTraits;

template <class Tag>
constexpr OpCategory op_category_v = OpTraits<Tag>::schema.category;

template <class Tag>
constexpr ArityKind op_arity_v = OpTraits<Tag>::schema.arity;

template <class Tag>
inline constexpr AritySpec op_inputs_v = OpTraits<Tag>::schema.inputs;

template <class Tag>
inline constexpr AritySpec op_outputs_v = OpTraits<Tag>::schema.outputs;

template <class Tag>
inline constexpr MutationKind op_mutation_v = OpTraits<Tag>::schema.mutation;

template <class Tag>
inline constexpr bool op_has_fixed_inputs_v =
    OpTraits<Tag>::schema.inputs.kind == ArityKind::Fixed;

template <class Tag>
inline constexpr bool op_has_fixed_outputs_v =
    OpTraits<Tag>::schema.outputs.kind == ArityKind::Fixed;

template <class Tag>
inline constexpr std::size_t op_num_inputs_v =
    OpTraits<Tag>::schema.inputs.arity;

template <class Tag>
inline constexpr std::size_t op_num_outputs_v =
    OpTraits<Tag>::schema.outputs.arity;

template <class Tag>
inline constexpr bool op_is_pure_v =
    OpTraits<Tag>::schema.mutation == MutationKind::OutOfPlace;

/// Comparison

template <> struct OpTraits<GreaterTag> {
   static constexpr std::string_view name = "GreaterThan";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<GreaterEqualTag> {
   static constexpr std::string_view name = "GreaterThanEqual";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<MaximumTag> {
   static constexpr std::string_view name = "Maximum";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

/// Ewise
template <> struct OpTraits<AddTag> {
   static constexpr std::string_view name = "Add";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<SubTag> {
   static constexpr std::string_view name = "Sub";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<MulTag> {
   static constexpr std::string_view name = "Mul";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<DivTag> {
   static constexpr std::string_view name = "Div";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<PowTag> {
   static constexpr std::string_view name = "Pow";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

/// LinAlg
template <> struct OpTraits<MatMulTag> {
   static constexpr std::string_view name = "MatMul";
   static constexpr OpSchema schema{
       .category = OpCategory::Contraction,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<SwapAxesTag> {
   static constexpr std::string_view name = "SwapAxes";
   static constexpr OpSchema schema{
       .category = OpCategory::Movement,
       .inputs = {ArityKind::Fixed, 2},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

/// Reduction
template <> struct OpTraits<SumTag> {
   static constexpr std::string_view name = "Sum";
   static constexpr OpSchema schema{
       .category = OpCategory::Reduction,
       .inputs = {ArityKind::Fixed, 1},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<MeanTag> {
   static constexpr std::string_view name = "Mean";
   static constexpr OpSchema schema{
       .category = OpCategory::Reduction,
       .inputs = {ArityKind::Fixed, 1},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

/// Transcendental
template <> struct OpTraits<SqrtTag> {
   static constexpr std::string_view name = "Sqrt";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseUnary,
       .inputs = {ArityKind::Fixed, 1},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<LogTag> {
   static constexpr std::string_view name = "Log";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseUnary,
       .inputs = {ArityKind::Fixed, 1},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<ExpTag> {
   static constexpr std::string_view name = "Exp";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseUnary,
       .inputs = {ArityKind::Fixed, 1},
       .outputs = {ArityKind::Fixed, 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

#endif // FUSION_CORE_OP_SCHEMA_OP_TRAITS_H