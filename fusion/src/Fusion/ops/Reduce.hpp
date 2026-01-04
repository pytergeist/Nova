// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef OPS_REDUCE_HPP
#define OPS_REDUCE_HPP

#include <string_view>
#include <vector>

#include "Fusion/core/ElementWise.hpp"
#include "Fusion/core/RawTensor.hpp"
#include "Fusion/core/Reduce.hpp"
#include "Fusion/cpu/SimdTags.hpp"
#include "Fusion/cpu/SimdTraits.hpp"

#include "Helpers.hpp"

namespace fusion {

namespace math {

template <typename T>
inline RawTensor<T> sum(const RawTensor<T> &x) { // TODO: This bypasses Tensor
   // buffer/boadcast in curr impl
   const T *y = x.get_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   return RawTensor<T>({1}, std::vector<T>{acc}, x.dtype(), x.device());
}

template <typename T> inline RawTensor<T> mean(const RawTensor<T> &x) {
   const T *y = x.get_ptr();
   const std::size_t n = x.flat_size();
   T acc = reduce::reduce_tag<T, GlobalSumSIMD>(y, n);
   T mean = acc / static_cast<T>(n);
   return RawTensor<T>({1}, std::vector<T>{acc}, x.dtype(), x.device());
}

} // namespace math

} // namespace fusion

#endif // OPS_REDUCE_HPP
