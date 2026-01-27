#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "TensorPlan.h"
// #include "PlanMeta.hpp"

using value_type = std::ptrdiff_t;

// ContractionPlan make_contraction_plan(const std::vector<TensorDescription>
// &descs) {
//    ContractionPlan plan;
//    plan.num_operands = descs.size();
//    plan.itemsize = descs[0].itemsize;
//    std::size_t max_ndims = 0;
//    for (const auto &desc : descs) {
//       max_ndims = std::max(desc.ndims, max_ndims);
//    }
//    plan.out_ndim = max_ndims - 1; // GeMM -> [M, K] @ [K, N] -> [M, N]
//
//    // This is currently hardcoded for matmul rules
//    std::int64_t lhs_contraction_axis = descs[0].ndims - 1;
//    std::int64_t rhs_contraction_axis = descs[0].ndims - 2;
//    // probably need to change the aboive to prebroadcast and recaculate post
//    plan
//
//    plan.caxes = ContractionAxes{.lhs_operand=lhs_contraction_axis,
//    .rhs_operand=rhs_contraction_axis};
//
//
//
//    // Hardcoded for two operands
//    std::vector<std::size_t> output_shape;
//    for (std::size_t i = 0; i < max_ndims; ++i) {
//       if (i < lhs_contraction_axis) {
//       output_shape.push_back(descs[0].shape[i]);
//       }
//       if (i > rhs_contraction_axis) {
//          output_shape.push_back(descs[0].shape[i]);
//       }
//    }
//    plan.out_shape = output_shape;
//    // here I am going to create a broadcast plan between out/lhs and out/rhs
//    TensorDescription dOut = make_desc_tmp(output_shape, nullptr);
//    std::vector<TensorDescription> full_descs{dOut, descs[0], descs[1]};
//    BroadcastPlan lhs_broadcast_plan = make_broadcast_plan(full_descs);
//
//    plan.loop = lhs_broadcast_plan.loop;
//
//    // This is currently hardcoded for matmul rules
//    std::int64_t lhs_contraction_axis_post_broadcast = descs[0].ndims - 1;
//    std::int64_t rhs_contraction_axis_post_broadcast = descs[0].ndims - 2;
//    // probably need to change the aboive to prebroadcast and recaculate post
//    plan plan.caxes =
//    ContractionAxes{.lhs_operand=lhs_contraction_axis_post_broadcast,
//    .rhs_operand=rhs_contraction_axis_post_broadcast};
//
//    // This is almost certainly not generic - test and validate
//    std::size_t contraction_loop_index = std::max(lhs_contraction_axis,
//    rhs_contraction_axis) - std::min(lhs_contraction_axis,
//    rhs_contraction_axis); LoopDim contraction_loop =
//    plan.loop[contraction_loop_index]; contraction_loop.stride_bytes[0] = 0;
//    plan.loop.erase(plan.loop.begin() + contraction_loop_index);
//    plan.loop.push_back(contraction_loop);
//
//
//    /*
//	The ordering of what is happening here is wrong:
//		- set a data member in loop called kind{indepedant, reduce,
//paired}
//		- create broadcastplan with {dOut, dA, dB}
//		- this create a common loop space
//		- remove loop for contraction axis from generic loop dim
//		- re-enter loop that is a pairwise loop as the final loop
//     */
//    return plan;
//
// }

inline void
validate_descs_same_itemsize(const std::vector<TensorDescription> &descs) {
   if (descs.empty())
      throw std::runtime_error("broadcast: no operands");
   const std::size_t itemsize = descs[0].itemsize;

   for (const TensorDescription &d : descs) {
      if (d.itemsize != itemsize)
         throw std::runtime_error(
             "You are trying to broadcast mixed datatype Tensors");
      if (d.ndims != d.shape.size() || d.ndims != d.strides.size()) {
         throw std::runtime_error(
             "broadcast: bad TensorDescription (ndims/shape/strides mismatch)");
      }
   }
}

inline std::size_t norm_axis(std::int64_t ax, std::size_t ndims) {
   const std::int64_t r =
       (ax < 0) ? (ax + static_cast<std::int64_t>(ndims)) : ax;
   if (r < 0 || r >= static_cast<std::int64_t>(ndims))
      throw std::runtime_error("axis out of range");
   return static_cast<std::size_t>(r);
}

inline std::size_t broadcast_dim(std::size_t a, std::size_t b) {
   if (a == b)
      return a;
   if (a == 1)
      return b;
   if (b == 1)
      return a;
   throw std::runtime_error("broadcast: dimension mismatch");
}

inline std::int64_t stride_bytes_for_binding(const TensorDescription &desc,
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

inline IndexSpaceIR
build_broadcast_ir_right_aligned(const std::vector<TensorDescription> &descs) {
   validate_descs_same_itemsize(descs);

   IndexSpaceIR ir;
   ir.num_operands = descs.size();
   ir.itemsize = descs[0].itemsize;

   std::size_t max_nd = 0;
   for (const TensorDescription &d : descs)
      max_nd = std::max(max_nd, d.ndims);

   ir.indices.resize(max_nd);
   ir.out_indices.resize(max_nd);

   for (std::size_t od = 0; od < max_nd; ++od) {
      IndexDef idx;
      idx.kind = IndexKind::Independent;
      idx.extent = 1;
      idx.axis_of_operand.assign(ir.num_operands, -1);

      for (std::size_t op = 0; op < ir.num_operands; ++op) {
         const TensorDescription &d = descs[op];
         const std::size_t pad = max_nd - d.ndims;

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

inline IndexSpaceIR
build_reduction_ir(const std::vector<TensorDescription> &descs,
                   std::size_t axis, bool keepdim) {
   validate_descs_same_itemsize(descs);
   if (descs.size() < 2)
      throw std::runtime_error("reduction: expected at least {out, in}");

   const TensorDescription &out_desc = descs[0];
   const TensorDescription &in_desc = descs.back();
   const std::size_t in_nd = in_desc.ndims;

   for (std::size_t op = 1; op < descs.size(); ++op) {
      if (descs[op].ndims != in_nd)
         throw std::runtime_error("reduction: input operand rank mismatch");
   }

   if (keepdim) {
      if (out_desc.ndims != in_nd)
         throw std::runtime_error(
             "reduction: keepdim expects out_ndims == in_ndims");
      if (out_desc.shape[axis] != 1)
         throw std::runtime_error(
             "reduction: keepdim expects out.shape[axis] == 1");
   } else {
      if (in_nd == 0)
         throw std::runtime_error(
             "reduction: cannot reduce scalar with keepdim=false");
      if (out_desc.ndims != in_nd - 1)
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

inline std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<TensorDescription> &descs,
               const std::vector<std::uint32_t> &loop_order) {
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
      ld.kind = (idx.kind == IndexKind::Reduction) ? LoopKind::Reduction
                                                   : LoopKind::Independent;
      ld.stride_bytes.resize(ir.num_operands);

      for (std::size_t op = 0; op < ir.num_operands; ++op) {
         if (op == 0 && idx.kind == IndexKind::Reduction) {
            ld.stride_bytes[op] = 0;
         } else {
            ld.stride_bytes[op] = stride_bytes_for_binding(
                descs[op], idx.axis_of_operand[op], idx.extent, ir.itemsize);
         }
      }

      loops.push_back(std::move(ld));
   }

   return loops;
}

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription> &descs) {
   IndexSpaceIR ir = build_broadcast_ir_right_aligned(descs);

   BroadcastPlan plan;
   plan.num_operands = ir.num_operands;
   plan.itemsize = ir.itemsize;

   plan.out_ndim = ir.out_indices.size();
   plan.out_shape.resize(plan.out_ndim);
   for (std::size_t i = 0; i < plan.out_ndim; ++i) {
      const std::uint32_t id = ir.out_indices[i];
      plan.out_shape[i] = ir.indices[id].extent;
   }

   const std::vector<std::uint32_t>& loop_order = ir.out_indices;

   plan.loop = lower_to_loops(ir, descs, loop_order);

   return plan;
}

ReductionPlan make_reduction_plan(const std::vector<TensorDescription> &descs,
                                  std::size_t axis, bool keepdim) {
   if (descs.size() < 2)
      throw std::runtime_error("reduction: expected at least {out, in}");

   const TensorDescription &in_desc = descs.back();
   const std::size_t in_nd = in_desc.ndims;
   const std::size_t ax = norm_axis(static_cast<std::int64_t>(axis), in_nd);

   IndexSpaceIR ir = build_reduction_ir(descs, ax, keepdim);

   ReductionPlan plan;
   plan.num_operands = descs.size();
   plan.itemsize = ir.itemsize;
   plan.keep_dim = keepdim;
   plan.reduction_axis = ax;

   plan.out_ndim = descs[0].ndims;
   plan.out_shape = descs[0].shape;

   std::vector<std::uint32_t> loop_order;
   loop_order.reserve(ir.indices.size());

   for (std::uint32_t id : ir.out_indices)
      loop_order.push_back(id);

   loop_order.push_back(static_cast<std::uint32_t>(ax));

   plan.loop = lower_to_loops(ir, descs, loop_order);

   return plan;
}



static IndexSpaceIR build_ir_from_einsum_binding(
    const std::vector<TensorDescription>& descs,
    const EinsumBinding& bind
) {
    validate_descs_same_itemsize(descs);

    if (bind.op_axis_labels.size() != descs.size()) {
        throw std::runtime_error("einsum: binding operand count mismatch");
    }

    IndexSpaceIR ir;
    ir.num_operands = descs.size();
    ir.itemsize = descs[0].itemsize;

    std::unordered_map<Label, std::uint32_t> label_to_id;
    label_to_id.reserve(64);

    auto ensure_label = [&](Label L) -> std::uint32_t {
        auto it = label_to_id.find(L);
        if (it != label_to_id.end()) return it->second;

        std::uint32_t id = static_cast<std::uint32_t>(ir.indices.size());
        IndexDef idx;
        idx.extent = 1;
        idx.kind = IndexKind::Reduction;
        idx.axis_of_operand.assign(ir.num_operands, -1);
        ir.indices.push_back(std::move(idx));
        label_to_id.emplace(L, id);
        return id;
    };

    for (std::size_t op = 0; op < descs.size(); ++op) {
        const auto& d = descs[op];
        const auto& labs = bind.op_axis_labels[op];

        if (labs.size() != d.ndims) {
            throw std::runtime_error("einsum: axis label count mismatch for operand");
        }

        {
            std::unordered_set<Label> seen;
            seen.reserve(labs.size());
            for (Label L : labs) {
                if (!seen.insert(L).second) {
                    throw std::runtime_error("einsum: repeated label within one operand (diagonal) not supported yet");
                }
            }
        }

        for (std::size_t ax = 0; ax < labs.size(); ++ax) {
            Label L = labs[ax];
            std::uint32_t id = ensure_label(L);
            IndexDef& idx = ir.indices[id];

            idx.axis_of_operand[op] = static_cast<std::int32_t>(ax);

            idx.extent = broadcast_dim(idx.extent, d.shape[ax]);
        }
    }

    ir.out_indices.clear();
    ir.out_indices.reserve(bind.out_labels.size());

    for (Label L : bind.out_labels) {
        auto it = label_to_id.find(L);
        if (it == label_to_id.end()) {
            throw std::runtime_error("einsum: output label does not appear in any operand");
        }
        const std::uint32_t id = it->second;
        ir.indices[id].kind = IndexKind::Independent;
        ir.out_indices.push_back(id);
    }

    {
        const auto& out_labs = bind.op_axis_labels[0];
        if (out_labs.size() != bind.out_labels.size()) {
            throw std::runtime_error("einsum: op0 labels must match out_labels length");
        }
        for (std::size_t i = 0; i < out_labs.size(); ++i) {
            if (out_labs[i] != bind.out_labels[i]) {
                throw std::runtime_error("einsum: op0 labels must equal out_labels (same order)");
            }
        }
    }

    return ir;
}

static std::vector<std::size_t> out_shape_from_ir(const IndexSpaceIR& ir) {
    std::vector<std::size_t> out_shape;
    out_shape.reserve(ir.out_indices.size());
    for (std::uint32_t id : ir.out_indices) {
        out_shape.push_back(ir.indices[id].extent);
    }
    return out_shape;
}

static inline std::int64_t stride_bytes_raw(const TensorDescription& d, std::int32_t axis, std::size_t itemsize) {
    if (axis < 0) return 0;
    return static_cast<std::int64_t>(d.strides[static_cast<std::size_t>(axis)]) *
           static_cast<std::int64_t>(itemsize);
}

static inline std::vector<std::int64_t> contig_elem_strides_local(const std::vector<std::size_t>& shape) {
    std::vector<std::int64_t> st(shape.size());
    std::int64_t r = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        st[static_cast<std::size_t>(i)] = r;
        r *= static_cast<std::int64_t>(shape[static_cast<std::size_t>(i)]);
    }
    return st;
}

static inline bool is_c_contiguous_local(const TensorDescription& d) {
    if (d.ndims == 0) return true;
    const auto expect = contig_elem_strides_local(d.shape);
    return d.strides == expect;
}

static inline std::size_t prod_extents(const IndexSpaceIR& ir, const std::vector<std::uint32_t>& ids) {
    std::size_t p = 1;
    for (auto id : ids) p *= ir.indices[id].extent;
    return p;
}

static inline LoopRole infer_role_for_id_two_inputs(const IndexDef& idx) {
    // Assumes operands: [0]=out, [1]=A, [2]=B
    const bool inA = (idx.axis_of_operand.size() > 1 && idx.axis_of_operand[1] >= 0);
    const bool inB = (idx.axis_of_operand.size() > 2 && idx.axis_of_operand[2] >= 0);

    if (idx.kind == IndexKind::Reduction) {
        return LoopRole::K;
    }

    if (inA && inB) return LoopRole::Batch;
    if (inA && !inB) return LoopRole::M;
    if (!inA && inB) return LoopRole::N;
    return LoopRole::Batch; // fallback
}

static inline std::vector<LoopDim>
lower_to_loops_with_roles(const IndexSpaceIR& ir,
                          const std::vector<TensorDescription>& descs,
                          const std::vector<std::uint32_t>& loop_order) {
    if (descs.size() != ir.num_operands)
        throw std::runtime_error("lower: desc count mismatch");

    std::vector<LoopDim> loops;
    loops.reserve(loop_order.size());

    for (std::uint32_t id : loop_order) {
        if (id >= ir.indices.size())
            throw std::runtime_error("lower: bad loop index id");

        const IndexDef& idx = ir.indices[id];

        LoopDim ld;
        ld.size = idx.extent;
        ld.kind = (idx.kind == IndexKind::Reduction) ? LoopKind::Reduction : LoopKind::Independent;
        ld.role = infer_role_for_id_two_inputs(idx);

        ld.stride_bytes.resize(ir.num_operands);

        for (std::size_t op = 0; op < ir.num_operands; ++op) {
            if (op == 0 && idx.kind == IndexKind::Reduction) {
                ld.stride_bytes[op] = 0;
            } else {
                ld.stride_bytes[op] = stride_bytes_for_binding(
                    descs[op], idx.axis_of_operand[op], idx.extent, ir.itemsize);
            }
        }

        loops.push_back(std::move(ld));
    }
    return loops;
}

static inline bool label_in_vec(Label L, const std::vector<Label>& v) {
    for (auto x : v) if (x == L) return true;
    return false;
}

static inline std::int32_t find_axis_of_label(const std::vector<Label>& labs, Label L) {
    for (std::size_t i = 0; i < labs.size(); ++i) {
        if (labs[i] == L) return static_cast<std::int32_t>(i);
    }
    return -1;
}

static inline std::vector<Label> labels_intersection(const std::vector<Label>& a, const std::vector<Label>& b) {
    std::vector<Label> out;
    out.reserve(std::min(a.size(), b.size()));
    for (auto x : a) {
        if (label_in_vec(x, b)) out.push_back(x);
    }
    return out;
}

static inline std::vector<Label> labels_difference(const std::vector<Label>& a, const std::vector<Label>& b) {
    std::vector<Label> out;
    out.reserve(a.size());
    for (auto x : a) {
        if (!label_in_vec(x, b)) out.push_back(x);
    }
    return out;
}

static inline bool try_make_gemm_like(const std::vector<TensorDescription>& descs,
                                     const EinsumBinding& binding,
                                     const IndexSpaceIR& ir,
                                     GemmLikeDesc& gemm_out) {
    if (descs.size() != 3) return false;

    const auto& out_labs = binding.out_labels;
    const auto& A_labs   = binding.op_axis_labels[1];
    const auto& B_labs   = binding.op_axis_labels[2];

    const auto AB_common = labels_intersection(A_labs, B_labs);
    const auto K_labs    = labels_difference(AB_common, out_labs);

    std::vector<Label> M_labs;
    for (auto L : out_labs) {
        if (label_in_vec(L, A_labs) && !label_in_vec(L, B_labs)) M_labs.push_back(L);
    }

    std::vector<Label> N_labs;
    for (auto L : out_labs) {
        if (label_in_vec(L, B_labs) && !label_in_vec(L, A_labs)) N_labs.push_back(L);
    }

    std::vector<Label> Batch_labs;
    for (auto L : out_labs) {
        if (label_in_vec(L, A_labs) && label_in_vec(L, B_labs)) Batch_labs.push_back(L);
    }

    if (M_labs.size() != 1 || N_labs.size() != 1 || K_labs.size() != 1) return false;

    const Label M = M_labs[0];
    const Label N = N_labs[0];
    const Label K = K_labs[0];

    const std::size_t batch_rank = Batch_labs.size();

    if (out_labs.size() != batch_rank + 2) return false;
    if (A_labs.size()   != batch_rank + 2) return false;
    if (B_labs.size()   != batch_rank + 2) return false;

    for (std::size_t i = 0; i < batch_rank; ++i) {
        if (out_labs[i] != Batch_labs[i]) return false;
        if (A_labs[i]   != Batch_labs[i]) return false;
        if (B_labs[i]   != Batch_labs[i]) return false;
    }

    if (!(out_labs[batch_rank] == M && out_labs[batch_rank + 1] == N)) return false;

    const bool A_mk = (A_labs[batch_rank] == M && A_labs[batch_rank + 1] == K);
    const bool A_km = (A_labs[batch_rank] == K && A_labs[batch_rank + 1] == M);
    const bool B_kn = (B_labs[batch_rank] == K && B_labs[batch_rank + 1] == N);
    const bool B_nk = (B_labs[batch_rank] == N && B_labs[batch_rank + 1] == K);

    if (!(A_mk || A_km)) return false;
    if (!(B_kn || B_nk)) return false;

    if (!is_c_contiguous_local(descs[0])) return false;
    if (!is_c_contiguous_local(descs[1])) return false;
    if (!is_c_contiguous_local(descs[2])) return false;

    for (std::size_t i = 0; i < batch_rank; ++i) {
        const auto ax_out = static_cast<std::int32_t>(i);
        const auto ax_A   = static_cast<std::int32_t>(i);
        const auto ax_B   = static_cast<std::int32_t>(i);

        const auto s_out = stride_bytes_for_binding(descs[0], ax_out, descs[0].shape[i], ir.itemsize);
        const auto s_A   = stride_bytes_for_binding(descs[1], ax_A,   descs[1].shape[i], ir.itemsize);
        const auto s_B   = stride_bytes_for_binding(descs[2], ax_B,   descs[2].shape[i], ir.itemsize);

        if (s_out == 0 || s_A == 0 || s_B == 0) return false;
    }

    const std::int32_t out_m_ax = find_axis_of_label(out_labs, M);
    const std::int32_t out_n_ax = find_axis_of_label(out_labs, N);
    const std::int32_t A_ax0    = static_cast<std::int32_t>(batch_rank);
    const std::int32_t A_ax1    = static_cast<std::int32_t>(batch_rank + 1);
    const std::int32_t B_ax0    = static_cast<std::int32_t>(batch_rank);
    const std::int32_t B_ax1    = static_cast<std::int32_t>(batch_rank + 1);

    const std::size_t M_extent = descs[0].shape[static_cast<std::size_t>(out_m_ax)];
    const std::size_t N_extent = descs[0].shape[static_cast<std::size_t>(out_n_ax)];

    const std::int32_t A_k_ax = find_axis_of_label(A_labs, K);
    if (A_k_ax < 0) return false;
    const std::size_t K_extent = descs[1].shape[static_cast<std::size_t>(A_k_ax)];

    std::size_t batch = 1;
    for (std::size_t i = 0; i < batch_rank; ++i) batch *= descs[0].shape[i];

    GemmLikeDesc g;
    g.batch = batch;
    g.M = M_extent;
    g.N = N_extent;
    g.K = K_extent;

    g.a_transpose = A_km;
    g.b_transpose = B_nk;

    g.out_rs = stride_bytes_raw(descs[0], out_m_ax, ir.itemsize);
    g.out_cs = stride_bytes_raw(descs[0], out_n_ax, ir.itemsize);

    g.a_rs = stride_bytes_raw(descs[1], A_ax0, ir.itemsize);
    g.a_cs = stride_bytes_raw(descs[1], A_ax1, ir.itemsize);

    g.b_rs = stride_bytes_raw(descs[2], B_ax0, ir.itemsize);
    g.b_cs = stride_bytes_raw(descs[2], B_ax1, ir.itemsize);

    g.out_is_contig_mn = true;
    g.a_is_contig_mk = A_mk;
    g.b_is_contig_kn = B_kn;

    gemm_out = g;
    return true;
}

ContractionPlan
make_contraction_plan_einsum(const std::vector<TensorDescription>& inputs,
                             const EinsumBinding& binding) {
    if (inputs.size() != 2) {
        throw std::runtime_error("einsum: current contraction planner expects exactly 2 inputs (A,B)");
    }
    validate_descs_same_itemsize(inputs);

    if (binding.op_axis_labels.size() != 3) {
        throw std::runtime_error("einsum: binding.op_axis_labels must be {out, A, B}");
    }
    if (binding.op_axis_labels[0].size() != binding.out_labels.size()) {
        throw std::runtime_error("einsum: op0 labels length must equal out_labels length");
    }

    TensorDescription dummy_out;
    dummy_out.ndims = binding.out_labels.size();
    dummy_out.shape.assign(dummy_out.ndims, 1);
    dummy_out.strides.assign(dummy_out.ndims, 0);
    dummy_out.itemsize = inputs[0].itemsize;

    std::vector<TensorDescription> descs;
    descs.reserve(3);
    descs.push_back(dummy_out);
    descs.push_back(inputs[0]);
    descs.push_back(inputs[1]);

    IndexSpaceIR ir = build_ir_from_einsum_binding(descs, binding);

    std::vector<std::size_t> out_shape = out_shape_from_ir(ir);

    TensorDescription out_desc;
    out_desc.ndims = out_shape.size();
    out_desc.shape = out_shape;
    out_desc.strides = contig_elem_strides_local(out_shape);
    out_desc.itemsize = inputs[0].itemsize;
    descs[0] = out_desc;

    std::vector<std::uint32_t> outer_order = ir.out_indices;

    std::vector<std::uint32_t> reduce_order;
    reduce_order.reserve(ir.indices.size());
    for (std::uint32_t id = 0; id < static_cast<std::uint32_t>(ir.indices.size()); ++id) {
        if (ir.indices[id].kind == IndexKind::Reduction) reduce_order.push_back(id);
    }

    std::vector<LoopDim> outer_loops = lower_to_loops_with_roles(ir, descs, outer_order);
    std::vector<LoopDim> reduce_loops = lower_to_loops_with_roles(ir, descs, reduce_order);

    ContractionPlan plan;
    plan.num_operands = descs.size();
    plan.itemsize = ir.itemsize;

    plan.out_ndim = out_shape.size();
    plan.out_shape = std::move(out_shape);

    plan.outer = std::move(outer_loops);
    plan.reduce = std::move(reduce_loops);

    plan.gemm_like = false;
    plan.gemm = GemmLikeDesc{};

    GemmLikeDesc g;
    if (try_make_gemm_like(descs, binding, ir, g)) {
        plan.gemm_like = true;
        plan.gemm = g;

        for (auto& ld : plan.outer) {
            (void)ld;
        }
        for (auto& ld : plan.reduce) {
            ld.role = LoopRole::K;
        }
    }

    return plan;
}
