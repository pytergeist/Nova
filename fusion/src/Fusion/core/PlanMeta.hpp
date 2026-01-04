#ifndef EWISE_META_HPP
#define EWISE_META_HPP

#include <vector>

#include "Broadcast.h"
#include "Reduction.h"

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
   bool keepdim;
   std::size_t reduction_axis;
   std::size_t reduce_len;
   TensorDescription dA, dOut;
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
inline TensorDescription make_desc(const std::vector<std::size_t> &shape,
                                   const int64_t *strides_elems) {
   // Create TensorDescription with ndims (shape.size()), int64_t vector of
   // sizes (shape), strides is stride_elems is not a nullptr, and itemsize
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

   auto dA = make_desc<T>(A.shape(), nullptr);
   auto dB = make_desc<T>(B.shape(), nullptr);
   auto plan_in = make_broadcast_plan({dA, dB});

   meta.fastpath = false;
   meta.out_shape.assign(plan_in.out_shape.begin(), plan_in.out_shape.end());
   meta.dOut = make_desc<T>(meta.out_shape, nullptr);
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

   auto dA = make_desc<T>(A.shape(), nullptr);
   auto plan_in = make_broadcast_plan({dA});

   meta.fastpath = false;
   meta.out_shape.assign(plan_in.out_shape.begin(), plan_in.out_shape.end());
   meta.dOut = make_desc<T>(meta.out_shape, nullptr);
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

   const TensorDescription dA = make_desc<T>(A.shape(), nullptr);

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
   meta.dOut = make_desc<T>(meta.out_shape, nullptr);
   meta.dA = dA;

   meta.plan = make_reduction_plan({meta.dOut, meta.dA}, axis, keepdim);
   meta.fastpath = false;
   meta.keepdim = keepdim;
   meta.reduction_axis = axis;
   meta.reduce_len = dA.shape[axis];

   return meta;
}

#endif // EWISE_META_HPP
