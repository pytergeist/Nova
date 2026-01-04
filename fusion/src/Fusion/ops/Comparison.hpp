// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef OPS_COMPARISON_HPP
#define OPS_COMPARISON_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/TensorIter.hpp"
#include "Fusion/core/PlanMeta.hpp"
#include "Fusion/core/RawTensor.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

template <typename T>
inline RawTensor<T> greater(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, GreaterThanSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> greater_equal(const RawTensor<T> &x,
                                  const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, GreaterThanEqualSIMD>(x, y, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> maximum(const RawTensor<T> &x, const RawTensor<T> &y) {
   BinaryEwiseMeta meta = make_binary_meta(x, y);
   RawTensor<T> out = init_out_from_meta(x, y, meta);
   fusion::iter::binary_ewise_tag<T, MaximumSIMD>(x, y, meta, out);
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_COMPARISON_HPP
