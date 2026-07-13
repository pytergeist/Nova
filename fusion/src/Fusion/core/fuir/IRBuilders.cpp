
#include "Fusion/core/fuir/IRBuilders.h"

#include <unordered_set>

#include "Fusion/core/fuir/ShapeRules.h"
#include "Fusion/core/fuir/FuirValidation.h"

namespace fusion::fuir {

namespace {

std::uint32_t
bind_idx_to_ir_by_label(std::unordered_map<Label, std::uint32_t> &label_to_id,
                        IndexSpaceIR &ir, Label L) {
   auto it = label_to_id.find(L);
   if (it != label_to_id.end())
      return it->second;

   std::uint32_t const id = static_cast<std::uint32_t>(ir.indices.size());
   IndexDef idx;
   idx.extent = 1;
   idx.label = L;
   idx.kind = IndexKind::Reduction;
   idx.axis_of_operand.assign(ir.num_operands, -1);
   ir.indices.push_back(std::move(idx));
   label_to_id.emplace(L, id);
   return id;
};

void set_out_labels_from_binding(
    std::unordered_map<Label, std::uint32_t> &label_to_id,
    const OperandLabelBinding &bind, IndexSpaceIR &ir) {
   for (Label L : bind.out_labels) {
      auto it = label_to_id.find(L);
      if (it == label_to_id.end()) {
         throw std::runtime_error(
             "einsum: output label does not appear in any operand");
      }
      const std::uint32_t id = it->second;
      ir.indices[id].kind = IndexKind::Independent;
      ir.out_indices.push_back(id);
   }
}

}

IndexSpaceIR
build_elementwise_ir_right_aligned(const std::vector<OperandDescription> &descs,
                                   const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_elementwise_ir_right_aligned";
   validate_descs_itemsize_group(descs, constraint, where);

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs[0].itemsize;

   std::size_t max_nd = 0;
   for (const OperandDescription &d : descs)
      max_nd = std::max(max_nd, d.ndims());

   ir.indices.resize(max_nd);
   ir.out_indices.resize(max_nd);

   for (std::size_t od = 0; od < max_nd; ++od) {
      IndexDef idx;
      idx.kind = IndexKind::Independent;
      idx.extent = 1;
      idx.axis_of_operand.assign(ir.num_operands, -1);

      for (std::size_t op = 0; op < ir.num_operands; ++op) {
         const OperandDescription &d = descs[op];
         const std::size_t pad = max_nd - d.ndims();

         if (od < pad) {
            idx.axis_of_operand[op] = -1;
            continue;
         }

         const std::size_t in_ax = od - pad; // right-aligned axis
         idx.axis_of_operand[op] = static_cast<std::int32_t>(in_ax);

         idx.extent = broadcast_dim(idx.extent, d.shape[in_ax]);
      }

      ir.indices[od] = std::move(idx);
      ir.out_indices[od] = static_cast<std::uint32_t>(od);
   }

   for (std::size_t od = 0; od < max_nd; ++od) {
      const IndexDef &idx = ir.indices[od];
      const std::size_t extent = idx.extent;

      for (std::size_t op = 0; op < ir.num_operands; ++op) {
         const std::int32_t ax = idx.axis_of_operand[op];
         if (ax < 0)
            continue;

         const std::size_t a = static_cast<std::size_t>(ax);
         const std::size_t dim_in = descs[op].shape[a];
         if (dim_in != 1 && dim_in != extent) {
            throw std::runtime_error(
                "broadcast: incompatible dimension (post extent)");
         }
      }
   }
   validate_index_space_ir(ir, where);
   return ir;
}

IndexSpaceIR build_reduction_ir(const std::vector<OperandDescription> &descs,
                                std::size_t axis, bool keepdim,
                                const OperandGroupConstraint constraint) {
   constexpr std::string_view where = "build_reduction_ir";

   validate_descs_itemsize_group(descs, constraint, where);
   validate_reduction_request(descs, axis, keepdim, where);

   if (descs.size() < 2)
      throw std::runtime_error("reduction: expected at least {out, in}");

   const OperandDescription &out_desc = descs[0];
   const OperandDescription &in_desc = descs.back();
   const std::size_t in_nd = in_desc.ndims();

   for (std::size_t op = 1; op < descs.size(); ++op) {
      if (descs[op].ndims() != in_nd)
         throw std::runtime_error("reduction: input operand rank mismatch");
   }

   if (keepdim) {
      if (out_desc.ndims() != in_nd)
         throw std::runtime_error(
             "reduction: keepdim expects out_ndims == in_ndims");
      if (out_desc.shape[axis] != 1)
         throw std::runtime_error(
             "reduction: keepdim expects out.shape[axis] == 1");
   } else {
      if (in_nd == 0)
         throw std::runtime_error(
             "reduction: cannot reduce scalar with keepdim=false");
      if (out_desc.ndims() != in_nd - 1)
         throw std::runtime_error("reduction: out_ndims must be in_ndims-1");
   }

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs[0].itemsize;

   ir.indices.resize(in_nd);

   ir.out_indices.clear();
   ir.out_indices.reserve(keepdim ? (in_nd - 1) : (in_nd - 1));

   auto out_axis_for_in_axis = [&](std::size_t in_ax) -> std::int32_t {
      if (keepdim) {
         return static_cast<std::int32_t>(in_ax);
      } else {
         if (in_ax == axis)
            return -1;
         if (in_ax < axis)
            return static_cast<std::int32_t>(in_ax);
         return static_cast<std::int32_t>(in_ax - 1);
      }
   };

   for (std::size_t in_ax = 0; in_ax < in_nd; ++in_ax) {
      IndexDef idx;
      idx.extent = in_desc.shape[in_ax];
      idx.kind =
          (in_ax == axis) ? IndexKind::Reduction : IndexKind::Independent;
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
   validate_index_space_ir(ir, where);
   return ir;
}

IndexSpaceIR build_ir_from_label_binding(const std::vector<OperandDescription> &descs,
                            const OperandLabelBinding &bind,
                            const OperandGroupConstraint constraint) {

   constexpr std::string_view where = "build_ir_from_label_binding";

   validate_descs_itemsize_group(descs, constraint, where);
   validate_operand_label_binding(descs, bind, where);

   if (bind.op_axis_labels.size() != descs.size()) {
      throw std::runtime_error("einsum: binding operand count mismatch");
   }

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs[0].itemsize; // NB: all operands must be same dtype

   std::unordered_map<Label, std::uint32_t> label_to_id;
   label_to_id.reserve(64);

   for (std::size_t op = 0; op < descs.size(); ++op) {
      const auto &d = descs[op];
      const auto &labs = bind.op_axis_labels[op];

      if (labs.size() != d.ndims()) {
         throw std::runtime_error(
             "einsum: axis label count mismatch for operand");
      }

      {
         std::unordered_set<Label> seen;
         seen.reserve(labs.size());
         for (Label L : labs) {
            if (!seen.insert(L).second) {
               throw std::runtime_error("einsum: repeated label within one "
                                        "operand (diagonal) not supported yet");
            }
         }
      }

      for (std::size_t ax = 0; ax < labs.size(); ++ax) {
         Label L = labs[ax];
         std::uint32_t id = bind_idx_to_ir_by_label(label_to_id, ir, L);
         IndexDef &idx = ir.indices[id];

         idx.axis_of_operand[op] = static_cast<std::int32_t>(ax);

         idx.extent = broadcast_dim(idx.extent, d.shape[ax]);
      }
   }

   ir.out_indices.clear();
   ir.out_indices.reserve(bind.out_labels.size());

   set_out_labels_from_binding(label_to_id, bind, ir);

   {
      const auto &out_labs = bind.op_axis_labels[0];
      if (out_labs.size() != bind.out_labels.size()) {
         throw std::runtime_error(
             "einsum: op0 labels must match out_labels length");
      }
      for (std::size_t i = 0; i < out_labs.size(); ++i) {
         if (out_labs[i] != bind.out_labels[i]) {
            throw std::runtime_error(
                "einsum: op0 labels must equal out_labels (same order)");
         }
      }
   }
   validate_index_space_ir(ir, where);
   return ir;
}

} // namespace fusion::fuir