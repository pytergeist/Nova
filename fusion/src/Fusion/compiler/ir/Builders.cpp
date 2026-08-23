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

#include <iostream>

namespace fusion::fuir {

namespace ferr = fusion::error;
using ferr::ErrorCategory;

namespace {

// calculate max_ndm

// Loop num operands:
// calculate logical indices/index kind from max ndim/padding
// create logical axis/set index kind

// set OperandId
// outputId = 0
// +1 A
// +2 B

// per OperandId:
// Set PhysicalAxisId
// i = 0
// j = i + 1
// k = j + 1
// ...

// Set extent
// extent = operand.shape[OperandAxis]

// Set AxisAccess
// Direct: if physical coord = logical coord | base + l * stride
// Indexed: if physical coord = I[logical coord] | base + I[l] * stride
// Broadcast: if physical coord = 0 | base + I[l] * 0

// create Axis Use
// create OperandUse

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
         extent = broadcast_dim(extent, desc.shape[rank - pad]);
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

struct AxisOccurrence {
   OperandId operand_id{0};
   PhysicalAxisId axis_id{0};
};

using LabelOccurrences = std::unordered_map<Label, std::vector<AxisOccurrence>>;

LabelOccurrences build_label_occurrences(const OperandLabelBinding &binding) {
   LabelOccurrences label_occurrences;
   for (std::size_t op = 0; op < binding.op_axis_labels.size(); ++op) {
      const std::vector<Label> &op_labels = binding.op_axis_labels[op];

      for (std::size_t ax = 0; ax < op_labels.size(); ++ax) {
         label_occurrences[op_labels[ax]].push_back(AxisOccurrence{
             .operand_id = static_cast<OperandId>(op),
             .axis_id = static_cast<PhysicalAxisId>(ax),
         });
      }
   }
   return label_occurrences;
}

std::vector<LogicalAxis>
build_contraction_logical_axes(const std::vector<OperandDescription> &descs,
                               const OperandLabelBinding &binding) {

   LabelOccurrences occurrences = build_label_occurrences(binding);

   std::vector<LogicalAxis> logical_axes;

   std::unordered_set<Label> label_set;

   for (std::size_t orank = 0; orank < binding.out_labels.size(); ++orank) {
      const Label &label = binding.out_labels[orank];
      const std::vector<AxisOccurrence> axis_occurrences =
          occurrences.at(label);
      if (!label_set.insert(label).second) {
         continue;
      }
      std::size_t extent = 1;
      for (const AxisOccurrence &axis_occurrence : axis_occurrences) {
         const PhysicalAxisId operand_id = axis_occurrence.operand_id;
         const PhysicalAxisId physical_axis_id = axis_occurrence.axis_id;
         const std::size_t physical_extent =
             descs.at(operand_id).shape[physical_axis_id];
         extent = broadcast_dim(extent, physical_extent);
      }
      logical_axes.emplace_back(LogicalAxis{
          .label = label, .extent = extent, .kind = IndexKind::Independent});
   }

   for (std::size_t op = 1; op < binding.op_axis_labels.size(); ++op) {
      const std::vector<Label> &op_labels = binding.op_axis_labels[op];

      for (const Label &label : op_labels) {
         if (std::ranges::find(binding.out_labels, label) !=
             binding.out_labels.end()) {
            continue;
         }
         const auto it = label_set.insert(label);
         if (!it.second) {
            continue;
         }

         const std::vector<AxisOccurrence> axis_occurrences =
             occurrences.at(label);
         const AxisOccurrence &first = axis_occurrences.front();
         const std::size_t extent =
             descs.at(first.operand_id).shape[first.axis_id];
         for (const AxisOccurrence &axis_occurrence : axis_occurrences) {
            const std::size_t cextent = descs.at(axis_occurrence.operand_id)
                                            .shape[axis_occurrence.axis_id];
            if (extent != cextent) {
               throw std::runtime_error("test");
            }
         }
         logical_axes.emplace_back(LogicalAxis{
             .label = label, .extent = extent, .kind = IndexKind::Reduction});
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

std::uint32_t
bind_idx_to_ir_by_label(std::unordered_map<Label, std::uint32_t> &label_to_id,
                        IndexSpaceIR &ir, const Label label) {
   const auto it = label_to_id.find(label);
   if (it != label_to_id.end()) {
      return it->second;
   }

   const std::uint32_t id = static_cast<std::uint32_t>(ir.indices.size());

   IndexDef idx;
   idx.extent = 1;
   idx.label = label;
   idx.kind = IndexKind::Reduction;
   idx.axis_of_operand.assign(ir.num_operands, -1);

   ir.indices.push_back(std::move(idx));
   label_to_id.emplace(label, id);

   return id;
}

void set_out_labels_from_binding(
    const std::unordered_map<Label, std::uint32_t> &label_to_id,
    const OperandLabelBinding &bind, IndexSpaceIR &ir,
    const std::string_view where) {
   for (const Label label : bind.out_labels) {
      const auto it = label_to_id.find(label);

      FUSION_INTERNAL_ASSERT_CODE(
          it != label_to_id.end(),
          fuir_error(FuirError::InvalidIR, ErrorCategory::Internal),
          ferr::message(
              where, ": fuir.binding.internal_missing_output_label: label ",
              label, " passed validation but was not present in label map"));

      const std::uint32_t id = it->second;
      ir.indices[id].kind = IndexKind::Independent;
      ir.out_indices.push_back(id);
   }
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

   ir.indices.resize(max_rank);
   ir.out_indices.resize(max_rank);
   ir.logical_axes = logical_axes;
   ir.physical_axes = physical_axes;
   ir.operand_use = operand_uses;

   for (std::size_t rank = 0; rank < max_rank; ++rank) {
      IndexDef idx;
      idx.label = static_cast<Label>(rank);
      idx.kind = IndexKind::Independent;
      idx.extent = 1;
      idx.axis_of_operand.assign(ir.num_operands, -1);

      for (std::size_t op = 0; op < ir.num_operands; ++op) {
         const OperandDescription &desc = descs[op];
         const std::size_t pad = max_rank - desc.ndims();

         if (rank < pad) {
            idx.axis_of_operand[op] = -1;
            continue;
         }

         const std::size_t in_ax = rank - pad;
         idx.axis_of_operand[op] = static_cast<std::int32_t>(in_ax);
         idx.extent = broadcast_dim(idx.extent, desc.shape[in_ax]);
      }

      ir.indices[rank] = std::move(idx);
      ir.out_indices[rank] = static_cast<std::uint32_t>(rank);
   }

   validation::validate_elementwise_index_space_ir(ir, where);
   return ir;
}

IndexSpaceIR build_reduction_ir(const std::vector<OperandDescription> &descs,
                                const std::size_t axis, const bool keepdim,
                                const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_reduction_ir";

   validation::validate_descs_itemsize_group(descs, constraint, where);
   validation::validate_reduction_request(descs, axis, keepdim, where);

   const OperandDescription &in_desc = descs.back();
   const std::size_t in_nd = in_desc.ndims();

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs.front().itemsize;

   ir.indices.resize(in_nd);
   ir.out_indices.clear();

   // TODO: keepdim reductions currently exclude the reduced axis from
   // out_indices. If IndexSpaceIR::out_indices becomes the sole source of
   // output shape truth, this needs to represent kept size-1 axes explicitly.
   ir.out_indices.reserve(in_nd > 0 ? in_nd - 1 : 0);

   const std::vector<LogicalAxis> logical_axes =
       build_unary_reduction_logical_axes(descs, axis);
   std::vector<std::vector<PhysicalAxis>> const physical_axes =
       build_operand_physical_axes(descs);
   std::vector<OperandUse> const operand_uses =
       build_unary_reduction_operand_uses(physical_axes, logical_axes, axis,
                                          keepdim);

   ir.logical_axes = logical_axes;
   ir.physical_axes = physical_axes;
   ir.operand_use = operand_uses;

   auto out_axis_for_in_axis = [&](const std::size_t in_ax) -> std::int32_t {
      if (keepdim) {
         return static_cast<std::int32_t>(in_ax);
      }

      if (in_ax == axis) {
         return -1;
      }

      if (in_ax < axis) {
         return static_cast<std::int32_t>(in_ax);
      }

      return static_cast<std::int32_t>(in_ax - 1);
   };

   for (std::size_t in_ax = 0; in_ax < in_nd; ++in_ax) {
      IndexDef idx;
      idx.label = static_cast<Label>(in_ax);
      idx.extent = in_desc.shape[in_ax];
      idx.kind = in_ax == axis ? IndexKind::Reduction : IndexKind::Independent;
      idx.axis_of_operand.assign(ir.num_operands, -1);

      idx.axis_of_operand[0] = out_axis_for_in_axis(in_ax);

      for (std::size_t op = 1; op < ir.num_operands; ++op) {
         idx.axis_of_operand[op] = static_cast<std::int32_t>(in_ax);
      }

      ir.indices[in_ax] = std::move(idx);

      if (in_ax != axis) {
         ir.out_indices.push_back(static_cast<std::uint32_t>(in_ax));
      }
   }
   validation::validate_unary_reduction_index_space_ir(ir, where);
   validation::validate_index_space_ir(ir, where);
   return ir;
}

std::string print_kind(const IndexKind kind) {
   if (kind == IndexKind::Independent) {
      return std::string("Independent");
   }
   if (kind == IndexKind::Reduction) {
      return std::string("Reduction");
   }
   return std::string("Unknown");
}

IndexSpaceIR
build_ir_from_label_binding(const std::vector<OperandDescription> &descs,
                            const OperandLabelBinding &bind,
                            const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_ir_from_label_binding";

   validation::validate_descs_itemsize_group(descs, constraint, where);
   validation::validate_operand_label_binding(descs, bind, where);

   const std::vector<LogicalAxis> logical_axes =
       build_contraction_logical_axes(descs, bind);
   const std::vector<std::vector<PhysicalAxis>> physical_axes =
       build_operand_physical_axes(descs);

   const std::vector<OperandUse> operand_uses =
       build_contraction_operand_uses(physical_axes, logical_axes, bind);

   // for (std::size_t ax = 0; ax < logical_axes.size(); ++ax) {
   //    std::cerr << "Logical Axis ID: " << ax << " | ";
   //    LogicalAxis axis = logical_axes[ax];
   //    std::cerr << "Label: " << axis.label << " | ";
   //    std::cerr << "Extent: " << axis.extent << " | ";
   //    std::cerr << "Kind: " << print_kind(axis.kind) << std::endl;
   // }

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs.front().itemsize;
   ir.logical_axes = logical_axes;
   ir.physical_axes = physical_axes;
   ir.operand_use = operand_uses;

   std::unordered_map<Label, std::uint32_t> label_to_id;
   label_to_id.reserve(64);

   for (std::size_t op = 0; op < descs.size(); ++op) {
      const OperandDescription &desc = descs[op];
      const std::vector<Label> &labels = bind.op_axis_labels[op];

      for (std::size_t ax = 0; ax < labels.size(); ++ax) {
         const Label label = labels[ax];

         const std::uint32_t id =
             bind_idx_to_ir_by_label(label_to_id, ir, label);

         IndexDef &idx = ir.indices[id];
         idx.axis_of_operand[op] = static_cast<std::int32_t>(ax);
         idx.extent = broadcast_dim(idx.extent, desc.shape[ax]);
      }
   }

   ir.out_indices.clear();
   ir.out_indices.reserve(bind.out_labels.size());

   set_out_labels_from_binding(label_to_id, bind, ir, where);

   validation::validate_index_space_ir(ir, where);
   return ir;
}

} // namespace fusion::fuir