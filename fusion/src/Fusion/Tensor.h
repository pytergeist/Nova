// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef TENSOR_H
#define TENSOR_H

#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/core/RawTensor.hpp"

template <typename T> using Tensor = ADTensor<T>;
template <typename T> using BaseTensor = RawTensor<T>;

#endif // TENSOR_H
