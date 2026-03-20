#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IR.h"

using value_type = std::ptrdiff_t;

void validate_descs_same_itemsize(
    const std::vector<OperandDescription> &descs) {
   if (descs.empty())
      throw std::runtime_error("broadcast: no operands");
   const std::size_t itemsize = descs[0].itemsize;

   for (const OperandDescription &d : descs) {
      if (d.itemsize != itemsize)
         throw std::runtime_error(
             "You are trying to broadcast mixed datatype Tensors");
      if (d.ndims() != d.shape.size() || d.ndims() != d.strides.size()) {
         throw std::runtime_error("broadcast: bad OperandDescription "
                                  "(ndims/shape/strides mismatch)");
      }
   }
}

std::size_t norm_axis(std::int64_t ax, std::size_t ndims) {
   const std::int64_t r =
       (ax < 0) ? (ax + static_cast<std::int64_t>(ndims)) : ax;
   if (r < 0 || r >= static_cast<std::int64_t>(ndims))
      throw std::runtime_error("axis out of range");
   return static_cast<std::size_t>(r);
}

std::size_t broadcast_dim(std::size_t a, std::size_t b) {
   if (a == b)
      return a;
   if (a == 1)
      return b;
   if (b == 1)
      return a;
   throw std::runtime_error("broadcast: dimension mismatch");
}

/// HERE

std::int64_t stride_bytes_for_binding(const OperandDescription &desc,
                                      std::int32_t axis,
                                      std::size_t index_extent,
                                      std::size_t itemsize) {
   if (axis < 0)
      return 0;

   const std::size_t ax = static_cast<std::size_t>(axis);
   const std::size_t dim_in = desc.shape[ax];

   if (dim_in == 1 && index_extent > 1)
      return 0;

   return static_cast<std::int64_t>(desc.strides[ax]) *
          static_cast<std::int64_t>(itemsize);
}

IndexSpaceIR
build_broadcast_ir_right_aligned(const std::vector<OperandDescription> &descs) {
   validate_descs_same_itemsize(descs);

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

   return ir;
}

IndexSpaceIR build_reduction_ir(const std::vector<OperandDescription> &descs,
                                std::size_t axis, bool keepdim) {
   validate_descs_same_itemsize(descs);
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

   return ir;
}

std::vector<OperandAccess>
lower_operand_access(const IndexSpaceIR &ir,
                     const std::vector<OperandDescription> &descs,
                     const std::vector<std::uint32_t> &loop_order) {
   std::vector<OperandAccess> op_access;
   op_access.reserve(ir.num_operands);

   // TODO: currently hardcoded to affine access

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
         if (loop_order[pos] >= ir.indices.size())
            throw std::runtime_error("lower: bad loop index id");

         if (const IndexDef &idx = ir.indices[loop_order[pos]];
             op == 0 && idx.kind == IndexKind::Reduction) {
            af.byte_stride_per_loop[pos] = 0;
         } else {
            af.byte_stride_per_loop[pos] = stride_bytes_for_binding(
                descs[op], idx.axis_of_operand[op], idx.extent, ir.itemsize);
         }
      }

      oa.affine = std::move(af);
      op_access.push_back(oa);
   }

   return op_access;
}

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<std::uint32_t> &loop_order) {
   /* TODO: Lower_to_loops currently doesn't set a IndexRole -- FIX */
   if (descs.size() != ir.num_operands)
      throw std::runtime_error("lower: desc count mismatch");

   std::vector<LoopDim> loops;
   loops.reserve(loop_order.size());

   for (std::uint32_t id : loop_order) {
      if (id >= ir.indices.size())
         throw std::runtime_error("lower: bad loop index id");

      const IndexDef &idx = ir.indices[id];

      LoopDim ld;
      ld.size = idx.extent;
      ld.kind = (idx.kind == IndexKind::Reduction) ? IndexKind::Reduction
                                                   : IndexKind::Independent;
      // ld.stride_bytes.resize(ir.num_operands);
      //
      // for (std::size_t op = 0; op < ir.num_operands; ++op) {
      //    if (op == 0 && idx.kind == IndexKind::Reduction) {
      //       ld.stride_bytes[op] = 0;
      //    } else {
      //       ld.stride_bytes[op] = stride_bytes_for_binding(
      //           descs[op], idx.axis_of_operand[op], idx.extent, ir.itemsize);
      //    }
      // }

      loops.push_back(std::move(ld));
   }

   return loops;
}

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<std::uint32_t> &loop_order,
               const std::vector<IndexRole> *role_of_id) {

   if (descs.size() != ir.num_operands)
      throw std::runtime_error("lower: desc count mismatch");

   std::vector<LoopDim> loops;
   loops.reserve(loop_order.size());

   for (std::uint32_t id : loop_order) {
      if (id >= ir.indices.size()) {
         throw std::runtime_error("lower: bad loop index id");
      }

      const IndexDef &idx = ir.indices[id];

      LoopDim ld;
      ld.size = idx.extent;
      ld.kind = (idx.kind == IndexKind::Reduction) ? IndexKind::Reduction
                                                   : IndexKind::Independent;

      if (role_of_id != nullptr) {
         ld.role = (*role_of_id)[id];
      } else {
         ld.role = IndexRole::Batch;
      }

      loops.push_back(std::move(ld));
   }

   return loops;
}

std::vector<IndexRole>
compute_roles_for_gemm_like(const IndexSpaceIR &ir,
                            const OperandLabelBinding &binding) {

   const auto &outL = binding.op_axis_labels[0];
   const auto &aL = binding.op_axis_labels[1];
   const auto &bL = binding.op_axis_labels[2];

   auto as_set = [](const std::vector<Label> &v) {
      std::unordered_set<Label> s;
      s.reserve(v.size());
      for (auto x : v)
         s.insert(x);
      return s;
   };

   auto outS = as_set(outL);
   auto aS = as_set(aL);
   auto bS = as_set(bL);

   std::unordered_map<Label, IndexRole> role_of_label;
   role_of_label.reserve(64);

   Label labelM = 0, labelN = 0, labelK = 0;
   bool haveM = false, haveN = false, haveK = false;

   std::unordered_set<Label> all;
   all.reserve(outS.size() + aS.size() + bS.size());
   for (auto x : outS)
      all.insert(x);
   for (auto x : aS)
      all.insert(x);
   for (auto x : bS)
      all.insert(x);

   for (Label L : all) {
      const bool inO = outS.count(L);
      const bool inA = aS.count(L);
      const bool inB = bS.count(L);

      if (inO && inA && inB) {
         role_of_label[L] = IndexRole::Batch;
      } else if (inO && inA && !inB) {
         role_of_label[L] = IndexRole::M;
         labelM = L;
         haveM = true;
      } else if (inO && !inA && inB) {
         role_of_label[L] = IndexRole::N;
         labelN = L;
         haveN = true;
      } else if (!inO && inA && inB) {
         role_of_label[L] = IndexRole::K;
         labelK = L;
         haveK = true;
      } else {
         role_of_label[L] = IndexRole::Batch;
      }
   }

   std::vector<IndexRole> role_of_id(ir.indices.size(), IndexRole::Batch);

   for (std::uint32_t id = 0; id < ir.indices.size(); ++id) {
      const Label L = ir.indices[id].label;
      auto it = role_of_label.find(L);
      if (it != role_of_label.end())
         role_of_id[id] = it->second;
   }

   return role_of_id;
}

std::uint32_t
bind_idx_to_ir_by_label(std::unordered_map<Label, std::uint32_t> &label_to_id,
                        IndexSpaceIR &ir, Label L) {
   auto it = label_to_id.find(L);
   if (it != label_to_id.end())
      return it->second;

   std::uint32_t id = static_cast<std::uint32_t>(ir.indices.size());
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
IndexSpaceIR
build_ir_from_label_binding(const std::vector<OperandDescription> &descs,
                            const OperandLabelBinding &bind) {
   // validate_descs_same_itemsize(descs);

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

   return ir;
}

std::vector<std::size_t> out_shape_from_ir(const IndexSpaceIR &ir) {
   std::vector<std::size_t> out_shape;
   out_shape.reserve(ir.out_indices.size());
   for (std::uint32_t id : ir.out_indices) {
      out_shape.push_back(ir.indices[id].extent);
   }
   return out_shape;
}

std::vector<std::size_t>
infer_einsum_out_shape(const std::vector<OperandDescription> &inputs,
                       const OperandLabelBinding &binding) {
   if (inputs.size() != 2) {
      throw std::runtime_error("einsum: expected inputs = {A, B}");
   }
   validate_descs_same_itemsize(inputs);

   OperandDescription dummy_out;
   dummy_out.shape.assign(binding.out_labels.size(), 1);
   dummy_out.strides.assign(dummy_out.ndims(), 0);
   dummy_out.itemsize = inputs[0].itemsize;

   std::vector<OperandDescription> tmp = {dummy_out, inputs[0], inputs[1]};
   IndexSpaceIR ir = build_ir_from_label_binding(tmp, binding);
   return out_shape_from_ir(ir);
}

std::int64_t stride_bytes_raw(const OperandDescription &d, std::int32_t axis,
                              std::size_t itemsize) {
   if (axis < 0)
      return 0;
   return static_cast<std::int64_t>(d.strides[static_cast<std::size_t>(axis)]) *
          static_cast<std::int64_t>(itemsize);
}

std::vector<std::int64_t>
contig_elem_strides_local(const std::vector<std::size_t> &shape) {
   std::vector<std::int64_t> st(shape.size());
   std::int64_t r = 1;
   for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
      st[static_cast<std::size_t>(i)] = r;
      r *= static_cast<std::int64_t>(shape[static_cast<std::size_t>(i)]);
   }
   return st;
}