#ifndef FUSION_CORE_PLANNING_OP_CONTEXT_BUILDERS_HPP
#define FUSION_CORE_PLANNING_OP_CONTEXT_BUILDERS_HPP

#include <cstddef>
#include <utility>
#include <vector>

#include "Fusion/core/fuir/IR.h"
#include "Fusion/core/planning/OpContext.h"
#include "Fusion/core/planning/OperandDescBuilders.h"
#include "Fusion/core/planning/PlanBuilders.h"

namespace fusion::planning {

template <typename T>
BinaryEwiseContext make_binary_ewise_context(const DenseTensor<T> &lhs,
                                             const DenseTensor<T> &rhs) {
   BinaryEwiseContext ctx{};

   const bool same_shape = lhs.shape() == rhs.shape();
   const bool both_contiguous = lhs.is_contiguous() && rhs.is_contiguous();

   if (same_shape && both_contiguous) {
      ctx.out_shape = lhs.shape();
      ctx.fast_len = lhs.flat_size();
      ctx.exec = BinaryExecKind::FlatContiguous;
      return ctx;
   }

   fuir::OperandDescription lhs_desc = make_desc_from_tensor<T>(lhs);
   fuir::OperandDescription rhs_desc = make_desc_from_tensor<T>(rhs);

   lhs_desc.update = fuir::UpdateKind::ReadOnly;
   rhs_desc.update = fuir::UpdateKind::ReadOnly;

   ElementwisePlan input_plan = make_elementwise_plan({lhs_desc, rhs_desc});

   ctx.out_shape.assign(input_plan.exec.core.out_shape.begin(),
                        input_plan.exec.core.out_shape.end());

   ctx.out = make_desc_from_shape<T>(ctx.out_shape, nullptr);
   ctx.out.update = fuir::UpdateKind::Overwrite;

   ctx.lhs = std::move(lhs_desc);
   ctx.rhs = std::move(rhs_desc);

   ctx.plan = make_elementwise_plan({ctx.out, ctx.lhs, ctx.rhs});

   const bool broadcast_lhs = ctx.lhs.shape != ctx.out.shape;

   ctx.exec = ctx.plan.exec.hints.all_contiguous_like
                  ? (broadcast_lhs ? BinaryExecKind::FlatContiguousBroadcastLHS
                                   : BinaryExecKind::FlatContiguousBroadcastRHS)
                  : BinaryExecKind::GenericStrided;

   return ctx;
}

template <typename T>
UnaryEwiseContext make_unary_ewise_context(const DenseTensor<T> &input) {
   UnaryEwiseContext ctx{};

   if (input.is_contiguous()) {
      ctx.fastpath = true;
      ctx.out_shape = input.shape();
      ctx.fast_len = input.flat_size();
      return ctx;
   }

   fuir::OperandDescription input_desc = make_desc_from_tensor<T>(input);
   input_desc.update = fuir::UpdateKind::ReadOnly;

   ElementwisePlan input_plan = make_elementwise_plan({input_desc});

   ctx.fastpath = false;
   ctx.out_shape.assign(input_plan.exec.core.out_shape.begin(),
                        input_plan.exec.core.out_shape.end());

   ctx.out = make_desc_from_shape<T>(ctx.out_shape, nullptr);
   ctx.out.update = fuir::UpdateKind::Overwrite;

   ctx.input = std::move(input_desc);
   ctx.plan = make_elementwise_plan({ctx.out, ctx.input});

   return ctx;
}

template <typename T>
ReductionContext make_reduction_context(const DenseTensor<T> &input,
                                        std::size_t axis, bool keepdim) {
   ReductionContext ctx{};

   if (axis == kGlobalReduceAxis && !keepdim) {
      ctx.fastpath = true;
      ctx.out_shape = std::vector<std::size_t>{1};
      ctx.fast_len = input.flat_size();
      ctx.reduce_len = ctx.fast_len;
      return ctx;
   }

   fuir::OperandDescription input_desc = make_desc_from_tensor<T>(input);
   input_desc.update = fuir::UpdateKind::ReadOnly;

   std::vector<std::size_t> out_shape;
   for (std::size_t d = 0; d < input_desc.ndims(); ++d) {
      if (d == axis) {
         if (keepdim) {
            out_shape.push_back(1);
         }
      } else {
         out_shape.push_back(input_desc.shape[d]);
      }
   }

   ctx.out_shape = std::move(out_shape);

   ctx.out = make_desc_from_shape<T>(ctx.out_shape, nullptr);
   ctx.out.update = fuir::UpdateKind::Accumulate;

   ctx.input = std::move(input_desc);

   ctx.plan = make_reduction_plan({ctx.out, ctx.input}, axis, keepdim);

   ctx.fastpath = false;
   ctx.keepdim = keepdim;
   ctx.reduction_axis = axis;
   ctx.reduce_len = ctx.input.shape[axis];

   return ctx;
}

template <typename T>
ContractionContext
make_contraction_context_einsum(const DenseTensor<T> &lhs,
                                const DenseTensor<T> &rhs,
                                const fuir::OperandLabelBinding &binding) {
   ContractionContext ctx{};

   ctx.lhs = make_desc_from_tensor<T>(lhs);
   ctx.rhs = make_desc_from_tensor<T>(rhs);

   ctx.lhs.update = fuir::UpdateKind::ReadOnly;
   ctx.rhs.update = fuir::UpdateKind::ReadOnly;

   ctx.out_shape = infer_out_shape_from_binding({ctx.lhs, ctx.rhs}, binding);

   ctx.out = make_desc_from_shape<T>(ctx.out_shape, nullptr);
   ctx.out.update = fuir::UpdateKind::Accumulate;

   ctx.plan =
       make_contraction_plan_einsum_out({ctx.out, ctx.lhs, ctx.rhs}, binding);

   ctx.fastpath = lhs.is_contiguous() && rhs.is_contiguous();
   ctx.fast_len = 0;
   ctx.binding = binding;

   return ctx;
}

} // namespace fusion::planning

#endif // FUSION_CORE_PLANNING_OP_CONTEXT_BUILDERS_HPP