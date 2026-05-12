#ifndef FUSION_CORE_FUIR_IR_H
#define FUSION_CORE_FUIR_IR_H

#include "DescContraints.h"
#include "Descs.h"

/// Symbolic identifier for a logical index (e.g. i, j, k in Einstein notation)
using Label = std::uint32_t;

enum class IndexKind { Independent, Reduction };

/// Role of a loop dimension in a contraction.
enum class IndexRole { Batch, M, N, K };

/// Describes a single logical index (loop dimension) in the index-space IR.
///
/// A logical index is a symbolic iteration dimension in a tensor expression,
/// analogous to `i`, `j`, or `k` in Einstein notation. Logical indices define
/// how operand axes relate to one another before lowering into a concrete
/// loop nest.
///
/// An IndexDef becomes one loop variable after lowering. It records:
/// - label: unique identifier for an axis, allows same IndexDef to be shared
/// across multiple operands.
/// - extent: the trip count for this index (after broadcasting / shape
/// unification)
/// - kind: whether the loop is Independent or Reduction (more to be added)
/// - axis_of_operand[op]: which axis of operand is bound to this index, or -1
/// if the operand does not depend on this index.
///
/// Invariants:
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
///      j: extent=3  kind=Independent  axis_of_operand=[1,1,1]   (A has dim 1
///      => broadcast, stride=0) k: extent=5  kind=Independent
///      axis_of_operand=[2,2,2]
///
/// 2) Reduction (y = sum(x, axis=1), keepdim=false)
///    x.shape = (4,5), y.shape = (4)
///    Operands: [0]=y, [1]=x
///      i: extent=4  kind=Independent  axis_of_operand=[0,0]
///      j: extent=5  kind=Reduction    axis_of_operand=[-1,1]    (reduction
///      axis absent from output)
///
/// 3) Contraction / einsum (C[i,j] = sum_k A[i,k] * B[k,j])
///    A.shape = (M,K), B.shape = (K,N), C.shape = (M,N)
///    Operands: [0]=C, [1]=A, [2]=B
///      i: extent=M  kind=Independent  axis_of_operand=[0,0,-1]
///      j: extent=N  kind=Independent  axis_of_operand=[1,-1,1]
///      k: extent=K  kind=Reduction    axis_of_operand=[-1,1,0]
///
/// NB:
/// - In (3), A does not depend on j and B does not depend on i, hence -1 for
/// those operands.
/// - The reduction index k has axis_of_operand[0] = -1 because the output has
/// no k axis.
///
struct IndexDef {
   Label label{0};
   std::size_t extent{1};
   IndexKind kind{IndexKind::Independent};
   std::vector<std::int32_t> axis_of_operand;
};

/// Describes the logical index space of a tensor expression.
///
/// IndexSpaceIR represents how logical indices relate across operands. It
/// records:
/// - num_operands: The number of operands participating in the expression
/// (input(s) + outputs(s))
/// - itemsize: Size in bytes of the operand data type.
/// - indices: List of logical indices participating in expression.
/// - out_indices: Subset of indices that define the output tensor shape.
///
/// Invariants:
/// - Currently, all operand must share the same dtype.
/// - Operand 0 is the output tensor.
/// - num_operands > 0.
struct IndexSpaceIR {
   std::size_t num_operands{0};
   std::size_t itemsize{0};

   std::vector<IndexDef> indices;
   std::vector<std::uint32_t> out_indices;
};

struct OperandLabelBinding {
   std::vector<std::vector<Label>> op_axis_labels;
   std::vector<Label> out_labels;
};

struct AffineAccess {
   std::vector<std::int64_t> byte_stride_per_loop;
};

struct BlockedAccess {};

struct IndexedAccess {};

/// Describes one lowered loop dimension in an execution plan.
///
/// LoopDim is produced by lowering the logical index from the IndexSpaceIR
/// into a concrete execution plan for the loop. It stores the trip count
/// for the loop and the per-operand byte-stride applied when the loop advances
/// by one iteration.
///
/// Invariants:
/// - Each LoopDim corresponds directly to one lowered logical index.
/// - The loop-dimension list is index-aligned: `loop_dims[i]` describes
///   lowered index `i`.
/// - Therefore, `loop_dims.size() == num_indices`.
struct LoopDim {
   // TODO: IndexKind and IndexRole are currently just set on init - need to add
   // set role/kind to lower_to_loop
   std::size_t size{};
   // std::vector<std::int64_t> stride_bytes;
   IndexKind kind{IndexKind::Independent};
   IndexRole role{IndexRole::Batch};
};


/// Describes the access pattern for a single operand in an execution plan.
///
/// OperandAccess is produced during lowering from an OperandDesc into a
/// concrete memory access description used by the executor. It captures
/// how an operand is laid out in memory, its ownership/update semantics,
/// and the addressing strategy used to compute element locations.
///
/// The active access descriptor is determined by `access`:
/// - AccessKind::Affine  -> `affine`
/// - AccessKind::Blocked -> `blocked`
/// - AccessKind::Indexed -> `indexed`
///
/// Together with the loop dimensions, OperandAccess defines how the
/// executor advances pointers and resolves addresses for each operand
/// during iteration.
///
/// Invariant:
/// - Each OperandAccess corresponds directly to one OperandDesc.
/// - The operand-access list is operand-id-aligned:
///   `operand_accesses[operand_id].operand_id == operand_id`.
/// - Therefore, `operand_accesses.size() == num_operands`.
struct OperandAccess {
   std::size_t operand_id{0};

   LayoutKind layout{LayoutKind::Dense}; // TODO: dense to generic, need to include contig for view support
   StorageKind storage{StorageKind::Owned};
   UpdateKind update{UpdateKind::ReadOnly};

   AccessKind access{AccessKind::Affine};

   AffineAccess affine{};
   BlockedAccess blocked{};
   IndexedAccess indexed{};
};

void validate_descs_same_itemsize(const std::vector<OperandDescription> &descs);

std::size_t norm_axis(std::int64_t ax, std::size_t ndims);

std::size_t broadcast_dim(std::size_t a, std::size_t b);

std::int64_t stride_bytes_for_binding(const OperandDescription &desc,
                                      std::int32_t axis,
                                      std::size_t index_extent,
                                      std::size_t itemsize);

IndexSpaceIR build_broadcast_ir_right_aligned(
    const std::vector<OperandDescription> &descs,
    const ItemSizeGroupConstraint constraint);

IndexSpaceIR
build_reduction_ir(const std::vector<OperandDescription> &descs,
                   std::size_t axis, bool keepdim,
                   const ItemSizeGroupConstraint constraint);

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<std::uint32_t> &loop_order);

std::vector<LoopDim>
lower_to_loops(const IndexSpaceIR &ir,
               const std::vector<OperandDescription> &descs,
               const std::vector<std::uint32_t> &loop_order,
               const std::vector<IndexRole> *role_of_id);

std::vector<OperandAccess>
lower_operand_access(const IndexSpaceIR &ir,
                     const std::vector<OperandDescription> &descs,
                     const std::vector<std::uint32_t> &loop_order);

std::vector<IndexRole>
compute_roles_for_gemm_like(const IndexSpaceIR &ir,
                            const OperandLabelBinding &binding);

IndexSpaceIR
build_ir_from_label_binding(const std::vector<OperandDescription> &descs,
                            const OperandLabelBinding &bind,
                            const ItemSizeGroupConstraint constraint);

std::vector<std::size_t> out_shape_from_ir(const IndexSpaceIR &ir);

std::vector<std::size_t>
infer_out_shape_from_binding(const std::vector<OperandDescription> &inputs,
                       const OperandLabelBinding &binding);

std::uint32_t
bind_idx_to_ir_by_label(std::unordered_map<Label, std::uint32_t> &label_to_id,
                        IndexSpaceIR &ir, Label L);

void set_out_labels_from_binding(
    std::unordered_map<Label, std::uint32_t> &label_to_id,
    const OperandLabelBinding &bind, IndexSpaceIR &ir);

std::int64_t stride_bytes_raw(const OperandDescription &d, std::int32_t axis,
                              std::size_t itemsize);

std::vector<std::int64_t>
contig_elem_strides_local(const std::vector<std::size_t> &shape);

#endif // FUSION_CORE_FUIR_IR_H