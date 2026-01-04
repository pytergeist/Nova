// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef BACKWARD_RESULT_HPP
#define BACKWARD_RESULT_HPP

#include <unordered_map>

#include "ADTypes.h"

#include "Fusion/core/RawTensor.hpp"

template <typename T> struct BackwardResult {
   std::unordered_map<std::int64_t, RawTensor<T>> grads;
   const bool empty() const { return grads.empty(); }
};

#endif // BACKWARD_RESULT_HPP
