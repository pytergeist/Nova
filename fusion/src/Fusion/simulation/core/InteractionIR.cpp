
#include <iostream>
#include <unordered_set>

#include "Fusion/core/fuir/IR.h"
#include "InteractionIR.h"

OperandLabelBinding make_gather_index_label_binding(std::size_t inp_nd,
                                                    std::size_t top_nd) {
   // if (out_nd < 1 || inp_nd < 2) {
   //    throw std::runtime_error(
   //       "Gather Index: expected rank >= 2 for both operands");
   //}

   // if (out_nd != inp_nd) {
   //  throw std::runtime_error("Gather Index: rank mismatch (input/output "
   //"operands must have the same dimensions");
   //}

   // Grab nd of input - batch index
   const std::size_t nd_c = inp_nd - 2;
   const Label base = 0;
   const Label Lc = static_cast<Label>(base + nd_c);
   const Label Ln = static_cast<Label>(base + nd_c + 1);
   const Label Le = static_cast<Label>(base + nd_c + 2);

   std::vector<Label> batch_labels(nd_c);
   std::size_t count = 0;
   for (std::size_t t = 0; t < nd_c; ++t) {
      batch_labels[t] = static_cast<Label>(base + t);
      count++;
   }

   std::vector<Label> top_labels = batch_labels;
   top_labels.push_back(Le);

   std::vector<Label> inp_labels = batch_labels;
   inp_labels.push_back(Lc);
   inp_labels.push_back(Ln);

   std::vector<Label> out_labels = batch_labels;
   out_labels.push_back(Lc);
   out_labels.push_back(Le);

   OperandLabelBinding binding;
   binding.op_axis_labels = {out_labels, inp_labels, top_labels};
   binding.out_labels = out_labels;
   return binding;
}
