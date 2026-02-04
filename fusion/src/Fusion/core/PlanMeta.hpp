#ifndef EWISE_META_HPP
#define EWISE_META_HPP

#include <vector>

#include "TensorPlan.h"

#include "RawTensor.hpp"

/* TODO: OPTIMIZE LATER: implament shape caching for broadcast plans, you need
 * to figure out what to cache and when. This will reduce the amount of plan
 * construction. Unodered_map impl? Also consider moving to a faster (poss
 * inlined) vec representation  */

template <typename T> struct RawTensor;

struct BinaryEwiseMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   BroadcastPlan plan;
   TensorDescription dA, dB, dOut;
};

struct UnaryEwiseMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   BroadcastPlan plan;
   TensorDescription dA, dOut;
};

struct ReductionMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   ReductionPlan plan;
   bool keepdim;               // TODO: Remove this it's also in the plan
   std::size_t reduction_axis; // TODO: This is also in the plan
   std::size_t reduce_len;
   TensorDescription dA, dOut;
};

struct ContractionMeta {
   bool fastpath;
   std::size_t fast_len;
   std::vector<std::size_t> out_shape;
   ContractionPlan plan;
   TensorDescription dA, dB, dOut;

   EinsumBinding binding;
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
inline TensorDescription
make_desc_from_shape(const std::vector<std::size_t> &shape,
                     const int64_t *strides_elems) {
   std::vector<std::size_t> sz(shape.begin(), shape.end());
   std::vector<std::int64_t> st;
   if (strides_elems) {
      st.assign(strides_elems,
                strides_elems + static_cast<int64_t>(shape.size()));
   } else {
      st = contig_elem_strides(shape);
   }
   return TensorDescription{static_cast<std::size_t>(shape.size()),
                            std::move(sz), std::move(st), sizeof(T)};
}

template <typename T>
static inline TensorDescription make_desc_from_tensor(const RawTensor<T> &t) {
   TensorDescription d;
   d.ndims = t.shape().size();
   d.shape = t.shape();
   d.itemsize = t.dtype_size();

   if constexpr (requires { t.strides(); }) {
      d.strides = t.strides();
   } else {
      d.strides = contig_elem_strides(d.shape);
   }
   return d;
}

template <typename T>
inline BinaryEwiseMeta make_binary_meta(const RawTensor<T> &A,
                                        const RawTensor<T> &B) {
   BinaryEwiseMeta meta{};
   const bool same = A.shape() == B.shape();
   const bool cont = A.is_contiguous() && B.is_contiguous();

   if (same && cont) {
      meta.fastpath = true;
      meta.out_shape = A.shape();
      meta.fast_len = A.flat_size();
      return meta;
   }
   // TODO: We are lying about strides here
   auto dA = make_desc_from_shape<T>(A.shape(), nullptr);
   auto dB = make_desc_from_shape<T>(B.shape(), nullptr);
   auto plan_in = make_broadcast_plan({dA, dB});

   meta.fastpath = false;
   meta.out_shape.assign(plan_in.out_shape.begin(), plan_in.out_shape.end());
   meta.dOut = make_desc_from_shape<T>(meta.out_shape, nullptr);
   meta.dA = std::move(dA);
   meta.dB = std::move(dB);
   meta.plan = make_broadcast_plan({meta.dOut, meta.dA, meta.dB});
   return meta;
};

template <typename T>
inline UnaryEwiseMeta make_unary_meta(const RawTensor<T> &A) {
   UnaryEwiseMeta meta{};
   const bool cont = A.is_contiguous();

   if (cont) {
      meta.fastpath = true;
      meta.out_shape = A.shape();
      meta.fast_len = A.flat_size();
      return meta;
   }
   // TODO: We are lying about strides here
   auto dA = make_desc_from_shape<T>(A.shape(), nullptr);
   auto plan_in = make_broadcast_plan({dA});

   meta.fastpath = false;
   meta.out_shape.assign(plan_in.out_shape.begin(), plan_in.out_shape.end());
   meta.dOut = make_desc_from_shape<T>(meta.out_shape, nullptr);
   meta.dA = std::move(dA);
   meta.plan = make_broadcast_plan({meta.dOut, meta.dA});
   return meta;
};

constexpr std::size_t kGlobalReduceAxis = -1;

template <typename T>
inline ReductionMeta make_reduction_meta(const RawTensor<T> &A,
                                         const std::size_t axis, bool keepdim) {
   ReductionMeta meta{};
   if (axis == kGlobalReduceAxis && keepdim == false) {
      meta.fastpath = true;
      meta.out_shape = std::vector<std::size_t>{1};
      meta.fast_len = A.flat_size();
      meta.reduce_len = meta.fast_len;
      return meta;
   }

   const TensorDescription dA = make_desc_from_shape<T>(A.shape(), nullptr);

   std::vector<std::size_t> out_shape;
   for (std::size_t d = 0; d < dA.ndims; ++d) {
      if (d == axis) {
         if (keepdim)
            out_shape.push_back(1);
      } else {
         out_shape.push_back(dA.shape[d]);
      }
   }
   meta.out_shape = out_shape;
   meta.dOut = make_desc_from_shape<T>(meta.out_shape, nullptr);
   meta.dA = dA;

   meta.plan = make_reduction_plan({meta.dOut, meta.dA}, axis, keepdim);
   meta.fastpath = false;
   meta.keepdim = keepdim;
   meta.reduction_axis = axis;
   meta.reduce_len = dA.shape[axis];

   return meta;
}

template <typename T>
inline ContractionMeta
make_contraction_meta_einsum(const RawTensor<T> &A, const RawTensor<T> &B,
                             const EinsumBinding &binding) {
   ContractionMeta meta{};

   // TODO: push real strides into here
   meta.dA = make_desc_from_tensor<T>(A);
   meta.dB = make_desc_from_tensor<T>(B);

   meta.out_shape = infer_einsum_out_shape({meta.dA, meta.dB}, binding);

   meta.dOut = make_desc_from_shape<T>(meta.out_shape, nullptr);

   meta.plan =
       make_contraction_plan_einsum_out({meta.dOut, meta.dA, meta.dB}, binding);

   meta.fastpath = A.is_contiguous() &&
                   B.is_contiguous(); // TODO: need better fast path here
   meta.fast_len = 0;
   meta.binding = binding;

   return meta;
}

#endif // EWISE_META_HPP
