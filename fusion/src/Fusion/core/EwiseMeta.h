#ifndef EWISE_META_H
#define EWISE_META_H

#include <vector>

#include "Broadcast.h"
#include "BroadcastIter.h"

#include "TensorBase.h"

template <typename T>
struct TensorBase;

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

template <typename T>
inline BinaryEwiseMeta make_binary_meta(const TensorBase<T>& A, const TensorBase<T>& B) {
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
    meta.out_shape.assign(plan_in.out_sizes.begin(), plan_in.out_sizes.end());
    meta.dOut = make_desc<T>(meta.out_shape, nullptr);
    meta.dA   = std::move(dA);
    meta.dB   = std::move(dB);
    meta.plan = make_broadcast_plan({meta.dOut, meta.dA, meta.dB});
    return meta;
};



template <typename T>
inline UnaryEwiseMeta make_binary_meta(const TensorBase<T>& A) {
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
    meta.out_shape.assign(plan_in.out_sizes.begin(), plan_in.out_sizes.end());
    meta.dOut = make_desc<T>(meta.out_shape, nullptr);
    meta.dA   = std::move(dA);
    meta.plan = make_broadcast_plan({meta.dOut, meta.dA});
    return meta;
};

#endif // EWISE_META_H
