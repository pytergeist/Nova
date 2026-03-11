
#include <iostream>
#include <unordered_set>

#include "PhysicsIR.h"

OperandLabelBinding make_gather_index_label_binding(std::size_t out_nd,
                                                    std::size_t inp_nd) {
   if (out_nd < 2 || inp_nd < 2) {
      throw std::runtime_error(
          "Gather Index: expected rank >= 2 for both operands");
   }

   if (out_nd != inp_nd) {
      throw std::runtime_error("Gather Index: rank mismatch (input/output "
                               "operands must have the same dimensions");
   }

   const std::size_t nd_c = out_nd - 2;
   std::cout << "nd_c: " << nd_c << std::endl;

   const Label base = 0;
   const Label Lc = static_cast<Label>(base + nd_c);
   const Label Le = static_cast<Label>(base + nd_c + 1);
   const Label Ln = static_cast<Label>(base + nd_c + 2);

   std::vector<Label> batch_labels(nd_c);
   std::size_t count = 0;
   for (std::size_t t = 0; t < nd_c; ++t) {
      batch_labels[t] = static_cast<Label>(base + t);
      count++;
   }

   std::vector<Label> out_labels = batch_labels;
   out_labels.push_back(Lc);
   out_labels.push_back(Le);

   std::vector<Label> inp_labels = batch_labels;
   inp_labels.push_back(Lc);
   inp_labels.push_back(Ln);

   OperandLabelBinding binding;
   binding.op_axis_labels = {out_labels, inp_labels};
   binding.out_labels = out_labels;
   return binding;
}

/// Builds IndexSpaceIR for gather and map operations
///
/// Reads dense/particle state through index map and writes
/// direct outputs. This output ir maps index space of
/// (c, e), where c = spatial dim, e = edges, to:
/// Y[c, e] = f(X[i(e), c], X[j(e), c])
///
/// Invariants:
/// - descs.size() == 2
/// - operand 0 is the output tensor, operand 1 in the .
/// - Output is accumulated $Δ_{c, e}$
/// - axis_of_operand.size() == IndexSpaceIR::num_operands.
///
IndexSpaceIR
build_gather_and_map_ir(const std::vector<OperandDescription> &descs) {
   // TODO: this axis_of_operand only expresses direct axis binding, we need
   // to introduce the concept of index binding/affine access to account for
   // i(e), j(e) behaviour of topological graphs
   if (descs.size() != 2) {
      throw std::runtime_error("Num operands > 2 for gather operation");
   }

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs[0].itemsize;

   std::size_t max_nd = 0;
   for (const OperandDescription &d : descs)
      max_nd = std::max(max_nd, d.ndims());

   std::unordered_map<Label, std::uint32_t> label_to_id;
   OperandLabelBinding binding =
       make_gather_index_label_binding(descs[0].ndims(), descs[1].ndims());

   for (std::size_t op = 0; op < descs.size(); ++op) {
      const auto &d = descs[op];
      const auto &labs = binding.op_axis_labels[op];

      if (labs.size() != d.ndims()) {
         throw std::runtime_error(
             "Binding: axis label count mismatch for operand");
      }

      for (std::size_t ax = 0; ax < labs.size(); ++ax) {
         Label L = labs[ax];
         std::uint32_t id = bind_idx_to_ir_by_label(label_to_id, ir, L);
         IndexDef &idx = ir.indices[id];

         idx.axis_of_operand[op] = static_cast<std::int32_t>(ax);

         idx.extent = d.shape[ax]; // TODO: evaluate this impl
      }
   }

   ir.out_indices.clear();
   ir.out_indices.reserve(binding.out_labels.size());

   set_out_labels_from_binding(label_to_id, binding, ir);

   return ir;
}