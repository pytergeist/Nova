#ifndef EWISE_META_HPP
#define EWISE_META_HPP

#include <vector>

#include "TensorPlan.h"

/* TODO: OPTIMIZE LATER: implament shape caching for broadcast plans, you need
 * to figure out what to cache and when. This will reduce the amount of plan
 * construction. Unodered_map impl? Also consider moving to a faster (poss
 * ) vec representation  */

template <typename T> struct DenseTensor;

template <typename T> class AoSoATensor;

enum class BinaryExecKind : std::uint8_t {
   GenericStrided,
   FlatContiguous,
   FlatContiguousBroadcastLHS,
   FlatContiguousBroadcastRHS,
};

struct BinaryEwiseMeta {
   std::vector<std::size_t> out_shape;
   std::size_t fast_len;
   BroadcastPlan plan;
   OperandDescription dA, dB, dOut;
   BinaryExecKind exec{BinaryExecKind::GenericStrided};
};

struct UnaryEwiseMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   BroadcastPlan plan;
   OperandDescription dA, dOut;
};

struct ReductionMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   ReductionPlan plan;
   bool keepdim;               // TODO: Remove this it's also in the plan
   std::size_t reduction_axis; // TODO: This is also in the plan
   std::size_t reduce_len;
   OperandDescription dA, dOut;
};

struct ContractionMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   ContractionPlan plan;
   OperandDescription dA, dB, dOut;

   OperandLabelBinding binding;
};

inline std::vector<std::int64_t>
contig_elem_strides(const std::vector<std::size_t> &shape) {
   std::vector<std::int64_t> st(shape.size());
   std::int64_t r = 1;
   for (int i = (int)shape.size() - 1; i >= 0; --i) {
      st[i] = r;
      r *= static_cast<std::int64_t>(shape[i]);
   }
   return st;
}

template <typename T>
OperandDescription make_desc_from_shape(const std::vector<std::size_t> &shape,
                                        const int64_t *strides_elems) {
   // TODO: Add IR update, access kind etc
   std::vector<std::size_t> sz(shape.begin(), shape.end());
   std::vector<std::int64_t> st;
   if (strides_elems) {
      st.assign(strides_elems,
                strides_elems + static_cast<int64_t>(shape.size()));
   } else {
      st = contig_elem_strides(shape);
   }
   return OperandDescription{std::move(sz), std::move(st), sizeof(T)};
}

template <typename T>
static OperandDescription make_desc_from_tensor(const DenseTensor<T> &t) {
   OperandDescription d;
   d.shape = t.shape();
   d.itemsize = t.dtype_size();

   if constexpr (requires { t.strides(); }) {
      d.strides = t.strides();
   } else {
      d.strides = contig_elem_strides(d.shape);
   }
   d.access = AccessKind::Affine;
   d.layout = t.is_contiguous() ? LayoutKind::Dense : LayoutKind::Strided;
   d.storage = !t.is_view() ? StorageKind::Owned : StorageKind::View;
   d.type = OperandDescType::Tensor;
   return d;
}

template <typename T>
static OperandDescription make_desc_from_aosoa_tensor(const AoSoATensor<T> &t) {
   OperandDescription d;
   d.shape = t.logical_shape();
   d.itemsize = t.raw().dtype_size(); // TODO: add forwarding?

   if constexpr (requires { t.strides(); }) {
      d.strides = t.strides();
   } else {
      d.strides = contig_elem_strides(d.shape);
   }
   d.access = AccessKind::Blocked;
   d.layout = LayoutKind::AoSoA;
   // TODO: Evaluate the below
   d.storage = StorageKind::Owned;
   d.type = OperandDescType::Tensor;
   return d;
}

inline std::string_view to_string(BinaryExecKind k) noexcept {
   switch (k) {
   case BinaryExecKind::GenericStrided:
      return "GenericStrided";
   case BinaryExecKind::FlatContiguous:
      return "FlatContiguous";
   case BinaryExecKind::FlatContiguousBroadcastLHS:
      return "FlatContiguousBroadcastLHS";
   case BinaryExecKind::FlatContiguousBroadcastRHS:
      return "FlatContiguousBroadcastRHS";
   }
   return "Unknown";
}

template <typename T>
BinaryEwiseMeta make_binary_meta(const DenseTensor<T> &A, const DenseTensor<T> &B) {

   BinaryEwiseMeta meta{};
   const bool same = A.shape() == B.shape();
   const bool cont = A.is_contiguous() && B.is_contiguous();

   if (same && cont) {
      meta.out_shape = A.shape();
      meta.fast_len = A.flat_size();
      meta.exec = BinaryExecKind::FlatContiguous;
      return meta;
   }

   OperandDescription dA = make_desc_from_tensor<T>(A);
   OperandDescription dB = make_desc_from_tensor<T>(B);

   dA.update = UpdateKind::ReadOnly;
   dB.update = UpdateKind::ReadOnly;

   BroadcastPlan plan_in = make_broadcast_plan({dA, dB});

   meta.out_shape.assign(plan_in.out_shape.begin(), plan_in.out_shape.end());
   meta.dOut = make_desc_from_shape<T>(meta.out_shape, nullptr);

   meta.dOut.update = UpdateKind::Overwrite;

   meta.dA = std::move(dA);
   meta.dB = std::move(dB);
   meta.plan = make_broadcast_plan({meta.dOut, meta.dA, meta.dB});

   bool const broadcastLHS{meta.dA.shape != meta.dOut.shape};

   // TODO: evaulate this impl - difficult for others to read
   meta.exec = meta.plan.all_contiguous_like
                   ? (broadcastLHS ? BinaryExecKind::FlatContiguousBroadcastLHS
                                   : BinaryExecKind::FlatContiguousBroadcastRHS)
                   : BinaryExecKind::GenericStrided;

   return meta;
};

template <typename T> UnaryEwiseMeta make_unary_meta(const DenseTensor<T> &A) {
   UnaryEwiseMeta meta{};
   const bool cont = A.is_contiguous();

   if (cont) { // TODO: this fastpath should also check is_view?
      meta.fastpath = true;
      meta.out_shape = A.shape();
      meta.fast_len = A.flat_size();
      return meta;
   }
   OperandDescription dA = make_desc_from_tensor<T>(A);
   dA.update = UpdateKind::ReadOnly;

   BroadcastPlan plan_in = make_broadcast_plan({dA});

   meta.fastpath = false;
   meta.out_shape.assign(plan_in.out_shape.begin(), plan_in.out_shape.end());
   meta.dOut = make_desc_from_shape<T>(meta.out_shape, nullptr);
   meta.dOut.update = UpdateKind::Overwrite;
   meta.dA = std::move(dA);
   meta.plan = make_broadcast_plan({meta.dOut, meta.dA});
   return meta;
};

constexpr std::size_t kGlobalReduceAxis = -1;

template <typename T>
ReductionMeta make_reduction_meta(const DenseTensor<T> &A, const std::size_t axis,
                                  bool keepdim) {
   ReductionMeta meta{};

   if (axis == kGlobalReduceAxis && !keepdim) {
      meta.fastpath = true;
      meta.out_shape = std::vector<std::size_t>{1};
      meta.fast_len = A.flat_size();
      meta.reduce_len = meta.fast_len;
      return meta;
   }

   OperandDescription dA = make_desc_from_tensor<T>(A);
   dA.update = UpdateKind::ReadOnly;

   std::vector<std::size_t> out_shape;
   for (std::size_t d = 0; d < dA.ndims(); ++d) {
      if (d == axis) {
         if (keepdim) {
            out_shape.push_back(1);
         }
      } else {
         out_shape.push_back(dA.shape[d]);
      }
   }
   meta.out_shape = out_shape;
   meta.dOut = make_desc_from_shape<T>(meta.out_shape, nullptr);
   meta.dOut.update = UpdateKind::Accumulate;
   meta.dA = std::move(dA);

   meta.plan = make_reduction_plan({meta.dOut, meta.dA}, axis, keepdim);
   meta.fastpath = false;
   meta.keepdim = keepdim;
   meta.reduction_axis = axis;
   meta.reduce_len = meta.dA.shape[axis];

   return meta;
}

template <typename T>
ContractionMeta
make_contraction_meta_einsum(const DenseTensor<T> &A, const DenseTensor<T> &B,
                             const OperandLabelBinding &binding) {
   ContractionMeta meta{};

   meta.dA = make_desc_from_tensor<T>(A);
   meta.dB = make_desc_from_tensor<T>(B);

   meta.dA.update = UpdateKind::ReadOnly;
   meta.dA.update = UpdateKind::ReadOnly;

   meta.out_shape = infer_out_shape_from_binding({meta.dA, meta.dB}, binding);

   meta.dOut = make_desc_from_shape<T>(meta.out_shape, nullptr);

   meta.dOut.update = UpdateKind::Accumulate;

   meta.plan =
       make_contraction_plan_einsum_out({meta.dOut, meta.dA, meta.dB}, binding);

   meta.fastpath = A.is_contiguous() &&
                   B.is_contiguous(); // TODO: need better fast path here
   meta.fast_len = 0;
   meta.binding = binding;

   return meta;
}

#endif // EWISE_META_HPP
