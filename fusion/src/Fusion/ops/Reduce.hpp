// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef OPS_REDUCE_HPP
#define OPS_REDUCE_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/RawTensor.hpp"
#include "Fusion/core/TensorIter.hpp"
#include "Fusion/cpu/SimdTags.hpp"
#include "Fusion/cpu/SimdTraits.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

template <typename T>
inline RawTensor<T> sum(const RawTensor<T> &x, const std::size_t axis,
                        const bool keep_dim) {
   ReductionMeta meta = make_reduction_meta(x, axis, keep_dim);
   RawTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::reduction_tag<T, SumSIMD>(x, meta, out);
   return out;
}

template <typename T>
inline RawTensor<T> mean(const RawTensor<T> &x, const std::size_t axis,
                         const bool keep_dim) {
   ReductionMeta meta = make_reduction_meta(x, axis, keep_dim);
   RawTensor<T> out = init_out_from_meta(x, meta);
   fusion::iter::reduction_tag<T, SumSIMD>(x, meta, out);
   const T norm_len = static_cast<T>(meta.reduce_len);
   out = out / norm_len;
   return out;
}

} // namespace math

} // namespace fusion

#endif // OPS_REDUCE_HPP
