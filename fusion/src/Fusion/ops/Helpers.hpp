// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef OP_HELPERS_HPP
#define OP_HELPERS_HPP

#include <cassert>

#include "Fusion/Tensor.h"
#include "Fusion/core/PlanMeta.hpp"

template <typename T>
inline RawTensor<T> init_out_from_meta(const RawTensor<T> &x,
                                       const RawTensor<T> &y,
                                       const BinaryEwiseMeta &m) {
   FUSION_CHECK(x.dtype() == y.dtype(), "dtypes do not match!");
   FUSION_CHECK(x.device() == y.device(), "devices do not match!");
   return RawTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
inline RawTensor<T> init_out_from_meta(const RawTensor<T> &x,
                                       const UnaryEwiseMeta &m) {
   return RawTensor<T>(m.out_shape, x.dtype(), x.device());
}

template <typename T>
inline RawTensor<T> init_out_from_meta(const RawTensor<T> &x,
                                       const ReductionMeta &m) {
   return RawTensor<T>(m.out_shape, x.dtype(), x.device());
}

#endif // OP_HELPERS_HPP
