#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "TensorPlan.h"
//#include "PlanMeta.hpp"

using value_type = std::ptrdiff_t;


//ContractionPlan make_contraction_plan(const std::vector<TensorDescription> &descs) {
//   ContractionPlan plan;
//   plan.num_operands = descs.size();
//   plan.itemsize = descs[0].itemsize;
//   std::size_t max_ndims = 0;
//   for (const auto &desc : descs) {
//      max_ndims = std::max(desc.ndims, max_ndims);
//   }
//   plan.out_ndim = max_ndims - 1; // GeMM -> [M, K] @ [K, N] -> [M, N]
//
//   // This is currently hardcoded for matmul rules
//   std::int64_t lhs_contraction_axis = descs[0].ndims - 1;
//   std::int64_t rhs_contraction_axis = descs[0].ndims - 2;
//   // probably need to change the aboive to prebroadcast and recaculate post plan
//
//   plan.caxes = ContractionAxes{.lhs_operand=lhs_contraction_axis, .rhs_operand=rhs_contraction_axis};
//
//
//
//   // Hardcoded for two operands
//   std::vector<std::size_t> output_shape;
//   for (std::size_t i = 0; i < max_ndims; ++i) {
//      if (i < lhs_contraction_axis) {
//      output_shape.push_back(descs[0].shape[i]);
//      }
//      if (i > rhs_contraction_axis) {
//         output_shape.push_back(descs[0].shape[i]);
//      }
//   }
//   plan.out_shape = output_shape;
//   // here I am going to create a broadcast plan between out/lhs and out/rhs
//   TensorDescription dOut = make_desc_tmp(output_shape, nullptr);
//   std::vector<TensorDescription> full_descs{dOut, descs[0], descs[1]};
//   BroadcastPlan lhs_broadcast_plan = make_broadcast_plan(full_descs);
//
//   plan.loop = lhs_broadcast_plan.loop;
//
//   // This is currently hardcoded for matmul rules
//   std::int64_t lhs_contraction_axis_post_broadcast = descs[0].ndims - 1;
//   std::int64_t rhs_contraction_axis_post_broadcast = descs[0].ndims - 2;
//   // probably need to change the aboive to prebroadcast and recaculate post plan
//   plan.caxes = ContractionAxes{.lhs_operand=lhs_contraction_axis_post_broadcast, .rhs_operand=rhs_contraction_axis_post_broadcast};
//
//   // This is almost certainly not generic - test and validate
//   std::size_t contraction_loop_index = std::max(lhs_contraction_axis, rhs_contraction_axis) - std::min(lhs_contraction_axis, rhs_contraction_axis);
//   LoopDim contraction_loop = plan.loop[contraction_loop_index];
//   contraction_loop.stride_bytes[0] = 0;
//   plan.loop.erase(plan.loop.begin() + contraction_loop_index);
//   plan.loop.push_back(contraction_loop);
//
//
//   /*
//	The ordering of what is happening here is wrong:
//		- set a data member in loop called kind{indepedant, reduce, paired}
//		- create broadcastplan with {dOut, dA, dB}
//		- this create a common loop space
//		- remove loop for contraction axis from generic loop dim
//		- re-enter loop that is a pairwise loop as the final loop
//    */
//   return plan;
//
//}

inline void validate_descs_same_itemsize(const std::vector<TensorDescription>& descs) {
    if (descs.empty()) throw std::runtime_error("broadcast: no operands");
    const std::size_t itemsize = descs[0].itemsize;

    for (const auto& d : descs) {
        if (d.itemsize != itemsize) throw std::runtime_error("broadcast: mixed itemsize");
        if (d.ndims != d.shape.size() || d.ndims != d.strides.size()) {
            throw std::runtime_error("broadcast: bad TensorDescription (ndims/shape/strides mismatch)");
        }
    }
}

inline std::size_t norm_axis(std::int64_t ax, std::size_t ndims) {
    const std::int64_t r = (ax < 0) ? (ax + static_cast<std::int64_t>(ndims)) : ax;
    if (r < 0 || r >= static_cast<std::int64_t>(ndims)) throw std::runtime_error("axis out of range");
    return static_cast<std::size_t>(r);
}


inline std::size_t broadcast_dim(std::size_t a, std::size_t b) {
    if (a == b) return a;
    if (a == 1) return b;
    if (b == 1) return a;
    throw std::runtime_error("broadcast: dimension mismatch");
}

inline std::int64_t stride_bytes_for_binding(
    const TensorDescription& desc,
    std::int32_t axis,            // -1 => operand does not depend on this index
    std::size_t index_extent,     // loop extent in the unified index space
    std::size_t itemsize
) {
    if (axis < 0) return 0;

    const std::size_t ax = static_cast<std::size_t>(axis);
    const std::size_t dim_in = desc.shape[ax];

    // Broadcasting: if operand axis is size-1 but index extent > 1, it doesn't advance.
    if (dim_in == 1 && index_extent > 1) return 0;

    // advance by operand stride * itemsize.
    return static_cast<std::int64_t>(desc.strides[ax]) * static_cast<std::int64_t>(itemsize);
}

inline IndexSpaceIR build_broadcast_ir_right_aligned(const std::vector<TensorDescription>& descs) {
    validate_descs_same_itemsize(descs);

    IndexSpaceIR ir;
    ir.num_operands = descs.size();
    ir.itemsize = descs[0].itemsize;

    // Determine max rank across operands (right alignment)
    std::size_t max_nd = 0;
    for (const auto& d : descs) max_nd = std::max(max_nd, d.ndims);

    ir.indices.resize(max_nd);
    ir.out_indices.resize(max_nd);

    for (std::size_t od = 0; od < max_nd; ++od) {
        IndexDef idx;
        idx.kind = IndexKind::Independent;
        idx.extent = 1;
        idx.axis_of_operand.assign(ir.num_operands, -1);

        // Compute binding + extent via broadcast rule
        for (std::size_t op = 0; op < ir.num_operands; ++op) {
            const auto& d = descs[op];
            const std::size_t pad = max_nd - d.ndims;

            if (od < pad) {
                // This is an implicit leading size-1 axis for this operand.
                idx.axis_of_operand[op] = -1;
                // extent unchanged (broadcast_dim with 1 would do nothing)
                continue;
            }

            const std::size_t in_ax = od - pad;              // right-aligned axis
            idx.axis_of_operand[op] = static_cast<std::int32_t>(in_ax);

            idx.extent = broadcast_dim(idx.extent, d.shape[in_ax]);
        }

        ir.indices[od] = std::move(idx);
        ir.out_indices[od] = static_cast<std::uint32_t>(od);
    }

    for (std::size_t od = 0; od < max_nd; ++od) {
        const auto& idx = ir.indices[od];
        const std::size_t extent = idx.extent;

        for (std::size_t op = 0; op < ir.num_operands; ++op) {
            const std::int32_t ax = idx.axis_of_operand[op];
            if (ax < 0) continue;

            const std::size_t a = static_cast<std::size_t>(ax);
            const std::size_t dim_in = descs[op].shape[a];
            if (dim_in != 1 && dim_in != extent) {
                throw std::runtime_error("broadcast: incompatible dimension (post extent)");
            }
        }
    }

    return ir;
}

inline IndexSpaceIR build_reduction_ir(
    const std::vector<TensorDescription>& descs,
    std::size_t axis,     // normalized axis in input rank
    bool keepdim
) {
    validate_descs_same_itemsize(descs);
    if (descs.size() < 2) throw std::runtime_error("reduction: expected at least {out, in}");

    const auto& out_desc = descs[0];
    const auto& in_desc  = descs.back();
    const std::size_t in_nd = in_desc.ndims;

    // Validate all input operands share the same rank as in_desc
    for (std::size_t op = 1; op < descs.size(); ++op) {
        if (descs[op].ndims != in_nd) throw std::runtime_error("reduction: input operand rank mismatch");
    }

    // Validate output rank matches keepdim policy
    if (keepdim) {
        if (out_desc.ndims != in_nd) throw std::runtime_error("reduction: keepdim expects out_ndims == in_ndims");
        if (out_desc.shape[axis] != 1) throw std::runtime_error("reduction: keepdim expects out.shape[axis] == 1");
    } else {
        if (in_nd == 0) throw std::runtime_error("reduction: cannot reduce scalar with keepdim=false");
        if (out_desc.ndims != in_nd - 1) throw std::runtime_error("reduction: out_ndims must be in_ndims-1");
    }

    IndexSpaceIR ir;
    ir.num_operands = descs.size();
    ir.itemsize = descs[0].itemsize;

    // One index per *input* axis.
    // This keeps the loop space identical to the input iteration space.
    ir.indices.resize(in_nd);

    // out_indices are the non-reduction indices in output axis order.
    ir.out_indices.clear();
    ir.out_indices.reserve(keepdim ? (in_nd - 1) : (in_nd - 1));

    auto out_axis_for_in_axis = [&](std::size_t in_ax) -> std::int32_t {
        if (keepdim) {
            // Output has same rank; each input axis maps to same output axis.
            return static_cast<std::int32_t>(in_ax);
        } else {
            // Output rank is one less; reduced axis is absent.
            if (in_ax == axis) return -1;
            if (in_ax < axis) return static_cast<std::int32_t>(in_ax);
            return static_cast<std::int32_t>(in_ax - 1);
        }
    };

    for (std::size_t in_ax = 0; in_ax < in_nd; ++in_ax) {
        IndexDef idx;
        idx.extent = in_desc.shape[in_ax];
        idx.kind = (in_ax == axis) ? IndexKind::Reduction : IndexKind::Independent;
        idx.axis_of_operand.assign(ir.num_operands, -1);

        // Bind output operand (op0)
        // - For reduced axis:
        //   keepdim=true  => bind to output axis (size 1); stride calc yields 0 anyway, and lowering forces 0.
        //   keepdim=false => axis absent => -1
        idx.axis_of_operand[0] = out_axis_for_in_axis(in_ax);

        // Bind each input operand axis directly to in_ax
        for (std::size_t op = 1; op < ir.num_operands; ++op) {
            idx.axis_of_operand[op] = static_cast<std::int32_t>(in_ax);
        }

        ir.indices[in_ax] = std::move(idx);

        if (in_ax != axis) {
            // Non-reduction indices appear in output in increasing axis order.
            ir.out_indices.push_back(static_cast<std::uint32_t>(in_ax));
        }
    }

    return ir;
}


// -------------------- Lower IR -> LoopDims (generic lowering) --------------------
//
// This is the reusable lowering for broadcast/reduction/contraction.
// For broadcast, all indices are Independent, so output strides behave normally.
//


inline std::vector<LoopDim> lower_to_loops(
    const IndexSpaceIR& ir,
    const std::vector<TensorDescription>& descs,
    const std::vector<std::uint32_t>& loop_order
) {
    if (descs.size() != ir.num_operands) throw std::runtime_error("lower: desc count mismatch");

    std::vector<LoopDim> loops;
    loops.reserve(loop_order.size());

    for (std::uint32_t id : loop_order) {
        if (id >= ir.indices.size()) throw std::runtime_error("lower: bad loop index id");

        const auto& idx = ir.indices[id];

        LoopDim ld;
        ld.size = idx.extent;
        ld.kind = (idx.kind == IndexKind::Reduction) ? LoopKind::Reduction : LoopKind::Independent;
        ld.stride_bytes.resize(ir.num_operands);

        for (std::size_t op = 0; op < ir.num_operands; ++op) {
            // Convention: output (op0) does not advance for reduction indices.
            // (For broadcast IR, idx.kind is Independent so this is a no-op.)
            if (op == 0 && idx.kind == IndexKind::Reduction) {
                ld.stride_bytes[op] = 0;
            } else {
                ld.stride_bytes[op] =
                    stride_bytes_for_binding(descs[op], idx.axis_of_operand[op], idx.extent, ir.itemsize);
            }
        }

        loops.push_back(std::move(ld));
    }

    return loops;
}


// -------------------- Public: make_broadcast_plan via IndexSpaceIR --------------------

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription>& descs) {
    // 1) Build IR
    IndexSpaceIR ir = build_broadcast_ir_right_aligned(descs);

    // 2) Derive out_shape from IR out_indices
    BroadcastPlan plan;
    plan.num_operands = ir.num_operands;
    plan.itemsize = ir.itemsize;

    plan.out_ndim = ir.out_indices.size();
    plan.out_shape.resize(plan.out_ndim);
    for (std::size_t i = 0; i < plan.out_ndim; ++i) {
        const auto id = ir.out_indices[i];
        plan.out_shape[i] = ir.indices[id].extent;
    }

    // 3) Choose loop order
    // For broadcast, output order is fine (later you can reorder for contiguity/vectorization)
    const auto loop_order = ir.out_indices;

    // 4) Lower IR -> loops
    plan.loop = lower_to_loops(ir, descs, loop_order);

    return plan;
}


// -------------------- Public: make_reduction_plan via IndexSpaceIR --------------------
ReductionPlan make_reduction_plan(
    const std::vector<TensorDescription>& descs,
    std::size_t axis,
    bool keepdim
) {
    if (descs.size() < 2) throw std::runtime_error("reduction: expected at least {out, in}");

    const auto& in_desc = descs.back();
    const std::size_t in_nd = in_desc.ndims;
    const std::size_t ax = norm_axis(static_cast<std::int64_t>(axis), in_nd);

    // 1) Build IR
    IndexSpaceIR ir = build_reduction_ir(descs, ax, keepdim);

    // 2) Fill plan metadata
    ReductionPlan plan;
    plan.num_operands = descs.size();
    plan.itemsize = ir.itemsize;
    plan.keep_dim = keepdim;
    plan.reduction_axis = ax;

    // Output shape is whatever the provided output desc says (important for keepdim layout)
    plan.out_ndim = descs[0].ndims;
    plan.out_shape = descs[0].shape;

    // 3) Choose loop order: independent (output) indices first, then reduction index last
    std::vector<std::uint32_t> loop_order;
    loop_order.reserve(ir.indices.size());

    // out_indices are input-axis ids for non-reduction axes
    for (auto id : ir.out_indices) loop_order.push_back(id);

    // reduction index is the input-axis id == ax
    loop_order.push_back(static_cast<std::uint32_t>(ax));

    // 4) Lower IR -> loops
    plan.loop = lower_to_loops(ir, descs, loop_order);

    return plan;
}
