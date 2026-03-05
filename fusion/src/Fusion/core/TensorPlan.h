#ifndef BROADCAST_ITERATOR_H
#define BROADCAST_ITERATOR_H

#include <cstddef>
#include <cstdint>
#include <vector>

/// Symbolic identifier for a logical index (e.g. i, j, k in Einstein notation)
using Label = std::uint32_t;

enum class IndexKind { Independent, Reduction };


/// Describes a single logical index (loop dimension) in the index-space IR.
///
/// An IndexDef becomes one loop variable after lowering. It records:
/// - label: unique identifier for an axis, allows same IndexDef to be shared across multiple operands.
/// - extent: the trip count for this index (after broadcasting / shape unification)
/// - kind: whether the loop is Independent or Reduction (more to be added)
/// - axis_of_operand[op]: which axis of operand is bound to this index, or -1 if the
///   operand does not depend on this index.
///
/// Conventions:
/// - operand 0 is the output tensor, operands 1...N are inputs.
/// - axis_of_operand.size() == IndexSpaceIR::num_operands.
/// - axis_of_operand[op] == -1 implies stride 0 for that operand on this loop.
///
/// Examples:
///
/// 1) Broadcast (C = A + B)
///    A.shape = (3,1,5), B.shape = (3,3,5), C.shape = (3,3,5)
///    Operands: [0]=C, [1]=A, [2]=B
///      i: extent=3  kind=Independent  axis_of_operand=[0,0,0]
///      j: extent=3  kind=Independent  axis_of_operand=[1,1,1]   (A has dim 1 => broadcast, stride=0)
///      k: extent=5  kind=Independent  axis_of_operand=[2,2,2]
///
/// 2) Reduction (y = sum(x, axis=1), keepdim=false)
///    x.shape = (4,5), y.shape = (4)
///    Operands: [0]=y, [1]=x
///      i: extent=4  kind=Independent  axis_of_operand=[0,0]
///      j: extent=5  kind=Reduction    axis_of_operand=[-1,1]    (reduction axis absent from output)
///
/// 3) Contraction / einsum (C[i,j] = sum_k A[i,k] * B[k,j])
///    A.shape = (M,K), B.shape = (K,N), C.shape = (M,N)
///    Operands: [0]=C, [1]=A, [2]=B
///      i: extent=M  kind=Independent  axis_of_operand=[0,0,-1]
///      j: extent=N  kind=Independent  axis_of_operand=[1,-1,1]
///      k: extent=K  kind=Reduction    axis_of_operand=[-1,1,0]
///
/// NB:
/// - In (3), A does not depend on j and B does not depend on i, hence -1 for those operands.
/// - The reduction index k has axis_of_operand[0] = -1 because the output has no k axis.
///
struct IndexDef {
   Label label{0};
   std::size_t extent{1};
   IndexKind kind{IndexKind::Independent};
   std::vector<std::int32_t> axis_of_operand;
};

/// Describes the logical index space of a tensor expression.
///
/// IndexSpaceIR represents how logical indices relate across operands. It records:
/// - num_operands: The number of operands participating in the expression (input(s) + outputs(s))
/// - itemsize: Size in bytes of the operand data type.
/// - indices: List of logical indices participating in expression.
/// - out_indices: Subset of indices that define the output tensor shape.
///
/// Conventions:
/// - Currently, all operand must share the same dtype.
/// - Operand 0 is the output tensor.
/// - num_operands > 0.
struct IndexSpaceIR {
   std::size_t num_operands{0};
   std::size_t itemsize{0};

   std::vector<IndexDef> indices;
   std::vector<std::uint32_t> out_indices;
};



struct EinsumBinding {
   std::vector<std::vector<Label>> op_axis_labels;
   std::vector<Label> out_labels;
};


struct TensorDescription {
   std::size_t ndims;
   std::vector<std::size_t> shape;
   std::vector<std::int64_t> strides;
   std::size_t itemsize;
};

enum class LoopKind { Independent, Reduction };
enum class LoopRole { Batch, M, N, K };

struct LoopDim {
   // TODO: LoopKind and LoopRole are currently just set on init - need to add
   // set role/kind to lower_to_loop
   std::size_t size;
   std::vector<std::int64_t> stride_bytes;
   LoopKind kind{LoopKind::Independent};
   LoopRole role{LoopRole::Batch};
};

struct BroadcastView {
   std::size_t out_ndim = 0;
   std::vector<std::size_t> out_shape;
   std::vector<std::vector<int>> axis_map;
   std::vector<std::vector<std::int64_t>> stride_bytes;
};

struct BroadcastPlan {
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;
   std::vector<LoopDim> loop;

   bool all_contiguous_like{false};
   std::size_t vector_bytes{0};

   std::size_t itemsize;
};

struct GemmLikeDesc {
   std::size_t batch{1};
   std::size_t M{1}, N{1}, K{1};

   std::int64_t out_rs{0}, out_cs{0};
   std::int64_t a_rs{0}, a_cs{0};
   std::int64_t b_rs{0}, b_cs{0};

   bool a_transpose{false};
   bool b_transpose{false};
   bool out_is_contig_mn{false};
   bool a_is_contig_mk{false};
   bool b_is_contig_kn{false};
};

struct ReductionPlan {
   std::size_t num_operands;
   std::size_t out_ndim;
   std::vector<std::size_t> out_shape;
   std::size_t reduction_axis;
   std::vector<LoopDim> loop;

   bool keep_dim{false};
   bool all_contiguous_like{false}; // curr not used - evaluate
   std::size_t vector_bytes{0};

   std::size_t itemsize;
};

struct ContractionPlan {
   std::size_t num_operands{0};
   std::size_t out_ndim{0};
   std::vector<std::size_t> out_shape;

   std::vector<LoopDim> loop;

   bool gemm_like{false};
   GemmLikeDesc gemm;

   std::size_t itemsize{0};
};

BroadcastPlan make_broadcast_plan(const std::vector<TensorDescription> &descs);

ReductionPlan make_reduction_plan(const std::vector<TensorDescription> &desc,
                                  const std::size_t axis, const bool keepdim);

ContractionPlan
make_contraction_plan_einsum(const std::vector<TensorDescription> &inputs,
                             const EinsumBinding &binding);

ContractionPlan
make_contraction_plan_einsum_out(const std::vector<TensorDescription> &descs,
                                 const EinsumBinding &binding);

std::vector<std::size_t>
infer_einsum_out_shape(const std::vector<TensorDescription> &inputs,
                       const EinsumBinding &binding);

#endif // BROADCAST_ITERATOR_H
