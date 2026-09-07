#include "Fusion/compiler/ir/Builders.h"

#include <algorithm>
#include <cstddef>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Fusion/common/error/Check.h"
#include "Fusion/compiler/ir/IRErrors.h"
#include "Fusion/compiler/ir/IRValidation.h"
#include "Fusion/compiler/ir/ShapeRules.h"


namespace fusion::fuir {

namespace ferr = fusion::error;
using ferr::ErrorCategory;

namespace {

std::vector<LogicalAxis>
build_elementwise_logical_axes(const std::vector<OperandDescription> &descs,
                               std::size_t max_rank) {
   std::vector<LogicalAxis> logical_axes;
   logical_axes.reserve(max_rank);
   for (std::size_t rank = 0; rank < max_rank; ++rank) {
      std::size_t extent = 1;
      for (const auto &desc : descs) {
         const std::size_t pad = max_rank - desc.ndims();
         if (rank < pad) {
            continue;
         }
         extent = shape::broadcast_dim(extent, desc.shape[rank - pad]);
      }
      logical_axes.emplace_back(LogicalAxis{
          .label = rank, .extent = extent, .kind = IndexKind::Independent});
   }
   return logical_axes;
}

std::vector<LogicalAxis>
build_unary_reduction_logical_axes(const std::vector<OperandDescription> &descs,
                                   const std::size_t axis) {
   const OperandDescription &in_desc = descs.back();

   std::vector<LogicalAxis> logical_axes;
   logical_axes.reserve(in_desc.ndims());

   for (std::size_t rank = 0; rank < in_desc.shape.size(); ++rank) {
      const IndexKind index_kind =
          axis == rank ? IndexKind::Reduction : IndexKind::Independent;
      logical_axes.emplace_back(LogicalAxis{
          .label = rank, .extent = in_desc.shape[rank], .kind = index_kind});
   }
   return logical_axes;
}

std::vector<LogicalAxis>
build_contraction_logical_axes(const shape::LabelExtentMap &label_extent_map,
                               const OperandLabelBinding &binding) {

   std::vector<LogicalAxis> logical_axes;
   logical_axes.reserve(label_extent_map.size());

   std::unordered_set<Label> labels_seen;
   labels_seen.reserve(label_extent_map.size());

   for (const Label label : binding.out_labels) {
      if (!labels_seen.insert(label).second) {
         continue;
      }
      logical_axes.emplace_back(
          LogicalAxis{.label = label,
                      .extent = label_extent_map.at(label),
                      .kind = IndexKind::Independent});
   }

   for (std::size_t op = 1; op < binding.op_axis_labels.size(); ++op) {
      for (const Label label : binding.op_axis_labels[op]) {
         if (!labels_seen.insert(label).second) {
            continue;
         }
         logical_axes.emplace_back(
             LogicalAxis{.label = label,
                         .extent = label_extent_map.at(label),
                         .kind = IndexKind::Reduction});
      }
   }
   return logical_axes;
}

using LogicalAxisIdByLabel = std::unordered_map<Label, LogicalAxisId>;

LogicalAxisIdByLabel
build_logical_axis_id_by_label(const std::vector<LogicalAxis> &logical_axes) {
   LogicalAxisIdByLabel ids_by_label;
   ids_by_label.reserve(logical_axes.size());

   for (std::size_t id = 0; id < logical_axes.size(); ++id) {
      const auto [it, inserted] = ids_by_label.emplace(
          logical_axes[id].label, static_cast<LogicalAxisId>(id));

      if (!inserted) {
         throw std::runtime_error("duplicate logical axis label");
      }
   }

   return ids_by_label;
}

std::size_t max_operand_rank(const std::vector<OperandDescription> &descs) {
   std::size_t max_nd = 0;
   for (const OperandDescription &desc : descs) {
      max_nd = std::max(max_nd, desc.ndims());
   }
   return max_nd;
}

std::vector<std::vector<PhysicalAxis>>
build_operand_physical_axes(const std::vector<OperandDescription> &descs) {
   std::vector<std::vector<PhysicalAxis>> physical_axes;
   physical_axes.resize(descs.size());
   for (std::size_t op = 0; op < descs.size(); ++op) {
      const OperandDescription &desc = descs[op];

      std::vector<PhysicalAxis> &axes = physical_axes[op];
      axes.reserve(desc.ndims());

      for (std::size_t ax = 0; ax < desc.ndims(); ++ax) {
         axes.emplace_back(
             PhysicalAxis{.operand_id = static_cast<OperandId>(op),
                          .axis_id = static_cast<std::uint32_t>(ax),
                          .extent = desc.shape[ax]});
      }
   }
   return physical_axes;
}

std::vector<OperandUse> build_unary_reduction_operand_uses(
    const std::vector<std::vector<PhysicalAxis>> &physical_axes,
    const std::vector<LogicalAxis> &logical_axes, const std::size_t axis,
    const bool keepdim) {
   std::vector<OperandUse> operand_uses;
   operand_uses.reserve(physical_axes.size());
   for (std::size_t op = 0; op < physical_axes.size(); ++op) {

      const std::vector<PhysicalAxis> &operand_axes = physical_axes[op];

      std::vector<AxisUse> axes_use;
      axes_use.reserve(operand_axes.size());

      for (const PhysicalAxis &physical_axis : operand_axes) {
         LogicalAxisId logical_axis_id;

         if (op == 0 && !keepdim) {
            logical_axis_id = static_cast<LogicalAxisId>(
                physical_axis.axis_id < axis ? physical_axis.axis_id
                                             : physical_axis.axis_id + 1);
         } else {
            logical_axis_id = static_cast<LogicalAxisId>(physical_axis.axis_id);
         }

         const LogicalAxis &logical_axis = logical_axes.at(logical_axis_id);

         const AxisAccess access =
             physical_axis.extent == 1 && logical_axis.extent > 1
                 ? AxisAccess::Broadcast
                 : AxisAccess::Direct;

         axes_use.emplace_back(AxisUse{
             .physical_axis_id = physical_axis.axis_id,
             .logical_axis_id = logical_axis_id,
             .access = access,
         });
      }

      operand_uses.emplace_back(OperandUse{
          .operand_id = static_cast<OperandId>(op),
          .axis_use = std::move(axes_use),
      });
   }

   return operand_uses;
}

std::vector<OperandUse> build_elementwise_operand_uses_right_aligned(
    const std::vector<std::vector<PhysicalAxis>> &physical_axes,
    const std::vector<LogicalAxis> &logical_axes, const std::size_t max_rank) {
   std::vector<OperandUse> operand_uses;
   operand_uses.reserve(physical_axes.size());
   for (std::size_t op = 0; op < physical_axes.size(); ++op) {

      const std::vector<PhysicalAxis> &operand_axes = physical_axes[op];
      const std::size_t pad = max_rank - operand_axes.size();

      std::vector<AxisUse> axes_use;
      axes_use.reserve(operand_axes.size());

      for (const PhysicalAxis &axis : operand_axes) {
         const LogicalAxisId logical_axis_id = static_cast<LogicalAxisId>(
             pad + static_cast<std::size_t>(axis.axis_id));
         const std::size_t logical_extent =
             logical_axes.at(logical_axis_id).extent;

         const AxisAccess access = axis.extent == 1 && logical_extent > 1
                                       ? AxisAccess::Broadcast
                                       : AxisAccess::Direct;

         axes_use.emplace_back(AxisUse{.physical_axis_id = axis.axis_id,
                                       .logical_axis_id = logical_axis_id,
                                       .access = access});
      }

      operand_uses.emplace_back(
          OperandUse{.operand_id = static_cast<OperandId>(op),
                     .axis_use = std::move(axes_use)});
   }
   return operand_uses;
};

std::vector<OperandUse> build_contraction_operand_uses(
    const std::vector<std::vector<PhysicalAxis>> &physical_axes,
    const std::vector<LogicalAxis> &logical_axes,
    const OperandLabelBinding &binding) {

   LogicalAxisIdByLabel logical_axis_id_by_label =
       build_logical_axis_id_by_label(logical_axes);
   std::vector<OperandUse> operand_uses;
   operand_uses.reserve(physical_axes.size());
   for (std::size_t op = 0; op < physical_axes.size(); ++op) {
      const std::vector<Label> &operand_labels = binding.op_axis_labels[op];
      const std::vector<PhysicalAxis> &operand_axes = physical_axes[op];

      std::vector<AxisUse> axes_use;
      axes_use.reserve(operand_axes.size());

      for (std::size_t ax = 0; ax < operand_axes.size(); ++ax) {
         const PhysicalAxis &physical_axis = operand_axes[ax];

         const Label label = operand_labels.at(physical_axis.axis_id);

         const LogicalAxisId logical_axis_id =
             logical_axis_id_by_label.at(label);

         const LogicalAxis &logical_axis = logical_axes.at(logical_axis_id);

         AxisAccess access = AxisAccess::Direct;

         if (physical_axis.extent != logical_axis.extent) {
            const bool valid_input_broadcast =
                op != 0 && logical_axis.kind == IndexKind::Independent &&
                physical_axis.extent == 1 && logical_axis.extent > 1;

            if (valid_input_broadcast) {
               access = AxisAccess::Broadcast;
            } else {
               throw std::runtime_error("invalid contraction axis mapping");
            }
         }

         axes_use.emplace_back(
             AxisUse{.physical_axis_id = physical_axis.axis_id,
                     .logical_axis_id = logical_axis_id,
                     .access = access});
      }

      operand_uses.emplace_back(
          OperandUse{.operand_id = static_cast<OperandId>(op),
                     .axis_use = std::move(axes_use)});
   }
   return operand_uses;
}

} // namespace

IndexSpaceIR
build_elementwise_ir_right_aligned(const std::vector<OperandDescription> &descs,
                                   const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_elementwise_ir_right_aligned";

   validation::validate_descs_itemsize_group(descs, constraint, where);

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs.front().itemsize;

   std::size_t max_rank = max_operand_rank(descs);

   std::vector<LogicalAxis> const logical_axes =
       build_elementwise_logical_axes(descs, max_rank);

   std::vector<std::vector<PhysicalAxis>> const physical_axes =
       build_operand_physical_axes(descs);

   const std::vector<OperandUse> operand_uses =
       build_elementwise_operand_uses_right_aligned(physical_axes, logical_axes,
                                                    max_rank);

   ir.logical_axes = logical_axes;
   ir.physical_axes = physical_axes;
   ir.operand_use = operand_uses;

   validation::validate_elementwise_index_space_ir(ir, where);
   return ir;
}

IndexSpaceIR build_reduction_ir(const std::vector<OperandDescription> &descs,
                                const std::size_t axis, const bool keepdim,
                                const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_reduction_ir";

   validation::validate_descs_itemsize_group(descs, constraint, where);
   validation::validate_reduction_request(descs, axis, keepdim, where);

   const std::vector<LogicalAxis> logical_axes =
       build_unary_reduction_logical_axes(descs, axis);
   std::vector<std::vector<PhysicalAxis>> const physical_axes =
       build_operand_physical_axes(descs);
   std::vector<OperandUse> const operand_uses =
       build_unary_reduction_operand_uses(physical_axes, logical_axes, axis,
                                          keepdim);

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs.front().itemsize;
   ir.logical_axes = logical_axes;
   ir.physical_axes = physical_axes;
   ir.operand_use = operand_uses;
   validation::validate_unary_reduction_index_space_ir(ir, where);
   return ir;
}

IndexSpaceIR
build_ir_from_label_binding(const std::vector<OperandDescription> &descs,
                            const OperandLabelBinding &binding,
                            const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_ir_from_label_binding";

   validation::validate_descs_itemsize_group(descs, constraint, where);
   validation::validate_operand_label_binding(descs, binding, where);

   const std::vector<OperandDescription> inputs(descs.begin() + 1, descs.end());
   const shape::LabelExtentMap label_extent_map =
       shape::resolve_contraction_label_extents(inputs, binding);

   const std::vector<LogicalAxis> logical_axes =
       build_contraction_logical_axes(label_extent_map, binding);
   const std::vector<std::vector<PhysicalAxis>> physical_axes =
       build_operand_physical_axes(descs);

   const std::vector<OperandUse> operand_uses =
       build_contraction_operand_uses(physical_axes, logical_axes, binding);

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs.front().itemsize;
   ir.logical_axes = logical_axes;
   ir.physical_axes = physical_axes;
   ir.operand_use = operand_uses;

   return ir;
}

} // namespace fusion::fuir