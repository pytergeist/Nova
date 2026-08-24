#include "Fusion/compiler/lowering/Lowering.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Fusion/compiler/ir/IRValidation.h"

namespace fusion::fuir {

// std::int64_t stride_bytes_for_binding(const OperandDescription &desc,
//                                       const std::int32_t axis,
//                                       const std::size_t index_extent,
//                                       const std::size_t itemsize) {
//    if (axis < 0) {
//       return 0;
//    }
//
//    const std::size_t ax = static_cast<std::size_t>(axis);
//    const std::size_t dim_in = desc.shape[ax];
//
//    if (dim_in == 1 && index_extent > 1) {
//       return 0;
//    }
//
//    return static_cast<std::int64_t>(desc.strides[ax]) *
//           static_cast<std::int64_t>(itemsize);
// }

std::int64_t stride_bytes_for_logical_axis(
    const OperandDescription &desc,
    const OperandUse &operand_use,
    const LogicalAxisId logical_axis_id,
    const std::size_t itemsize) {
   for (const AxisUse &axis_use : operand_use.axis_use) {
      if (axis_use.logical_axis_id != logical_axis_id) {
         continue;
      }

      switch (axis_use.access) {
      case AxisAccess::Direct:
         return static_cast<std::int64_t>(
                    desc.strides.at(axis_use.physical_axis_id)) *
                static_cast<std::int64_t>(itemsize);

      case AxisAccess::Broadcast:
         return 0;

      // case AxisAccess::Indexed:
      //    FUSION_INTERNAL_ASSERT_CODE(
      //        false,
      //        fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
      //        ferr::message(
      //            where,
      //            ": fuir.lowering.indexed_axis_use_in_affine_lowering: "
      //            "operand ",
      //            operand_use.operand_id, " maps logical_axis_id ",
      //            logical_axis_id,
      //            " using indexed access, which cannot be represented by an "
      //            "affine byte stride"));
      //    return 0;
      }
   }

   return 0;
}

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<std::uint32_t> &loop_order) {
   constexpr std::string_view where = "lower_to_loops";

   validation::validate_ir_matches_descs(ir, descs, where);
   validation::validate_loop_order(ir, loop_order, where);

   std::vector<LoopDim> loops;
   loops.reserve(loop_order.size());

   for (const std::uint32_t id : loop_order) {
      const LogicalAxis &axis = ir.logical_axes.at(id);

      LoopDim ld;
      ld.size = axis.extent;
      ld.kind = axis.kind;
      ld.role = IndexRole::Batch;

      loops.push_back(std::move(ld));
   }

   return loops;
}

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<std::uint32_t> &loop_order,
               const std::vector<IndexRole> *role_of_id) {
   constexpr std::string_view where = "lower_to_loops";

   validation::validate_ir_matches_descs(ir, descs, where);
   validation::validate_loop_order(ir, loop_order, where);
   validation::validate_role_vector_matches_ir(ir, role_of_id, where);

   std::vector<LoopDim> loops;
   loops.reserve(loop_order.size());

   for (const std::uint32_t id : loop_order) {
      const LogicalAxis &axis = ir.logical_axes.at(id);

      LoopDim ld;
      ld.size = axis.extent;
      ld.kind = axis.kind;
      ld.role = role_of_id != nullptr ? (*role_of_id)[id] : IndexRole::Batch;

      loops.push_back(std::move(ld));
   }

   return loops;
}

std::vector<OperandAccess>
lower_operand_access(const IndexSpaceIR &ir,
                     const std::vector<OperandDescription> &descs,
                     const std::vector<std::uint32_t> &loop_order) {
   constexpr std::string_view where = "lower_operand_access";

   validation::validate_ir_matches_descs(ir, descs, where);
   validation::validate_loop_order(ir, loop_order, where);

   std::vector<OperandAccess> op_access;
   op_access.reserve(ir.num_operands);

   for (std::size_t op = 0; op < ir.num_operands; ++op) {
      OperandAccess oa;
      AffineAccess af;

      oa.operand_id = op;
      oa.layout = descs[op].layout;
      oa.storage = descs[op].storage;
      oa.update = descs[op].update;
      oa.access = descs[op].access;

      af.byte_stride_per_loop.resize(loop_order.size());

      for (std::size_t pos = 0; pos < loop_order.size(); ++pos) {
         const std::uint32_t index_id = loop_order[pos];
         const LogicalAxis &axis = ir.logical_axes.at(index_id);

         if (op == 0 && axis.kind == IndexKind::Reduction) {
            af.byte_stride_per_loop[pos] = 0;
            continue;
         }

         af.byte_stride_per_loop[pos] = stride_bytes_for_logical_axis(
             descs.at(op), ir.operand_use.at(op), index_id, ir.itemsize);
      }

      oa.affine = std::move(af);
      op_access.push_back(std::move(oa));
   }

   return op_access;
}

std::vector<IndexRole>
compute_roles_for_gemm_like(const IndexSpaceIR &ir,
                            const OperandLabelBinding &binding) {
   validation::validate_index_space_ir(ir, "compute_roles_for_gemm_like");

   const auto &outL = binding.op_axis_labels[0];
   const auto &aL = binding.op_axis_labels[1];
   const auto &bL = binding.op_axis_labels[2];

   auto as_set = [](const std::vector<Label> &values) {
      std::unordered_set<Label> result;
      result.reserve(values.size());

      for (Label label : values) {
         result.insert(label);
      }

      return result;
   };

   const auto outS = as_set(outL);
   const auto aS = as_set(aL);
   const auto bS = as_set(bL);

   std::unordered_map<Label, IndexRole> role_of_label;
   role_of_label.reserve(64);

   std::unordered_set<Label> all;
   all.reserve(outS.size() + aS.size() + bS.size());

   for (Label label : outS) {
      all.insert(label);
   }

   for (Label label : aS) {
      all.insert(label);
   }

   for (Label label : bS) {
      all.insert(label);
   }

   for (Label label : all) {
      const bool inO = outS.contains(label);
      const bool inA = aS.contains(label);
      const bool inB = bS.contains(label);

      if (inO && inA && inB) {
         role_of_label[label] = IndexRole::Batch;
      } else if (inO && inA && !inB) {
         role_of_label[label] = IndexRole::M;
      } else if (inO && !inA && inB) {
         role_of_label[label] = IndexRole::N;
      } else if (!inO && inA && inB) {
         role_of_label[label] = IndexRole::K;
      } else {
         role_of_label[label] = IndexRole::Batch;
      }
   }

   std::vector<IndexRole> role_of_id(ir.logical_axes.size(), IndexRole::Batch);

   for (std::uint32_t id = 0;
        id < static_cast<std::uint32_t>(ir.logical_axes.size()); ++id) {
      const Label label = ir.logical_axes[id].label;

      if (const auto it = role_of_label.find(label);
          it != role_of_label.end()) {
         role_of_id[id] = it->second;
      }
   }

   return role_of_id;
}

} // namespace fusion::fuir