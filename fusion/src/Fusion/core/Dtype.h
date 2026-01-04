// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef DTYPE_HPP
#define DTYPE_HPP

#include <cstddef>

enum class DType {
   FLOAT32 = 0,
   FLOAT64 = 1,
   INT32 = 2,
   INT64 = 3,
   BOOL = 4,
};

constexpr DType kFloat32 = DType::FLOAT32;
constexpr DType kFloat64 = DType::FLOAT64;
constexpr DType kInt32 = DType::INT32;
constexpr DType kInt64 = DType::INT64;
constexpr DType kBool = DType::BOOL;

inline std::size_t get_dtype_size(DType dtype) {
   switch (dtype) {
   case DType::FLOAT32:
      return sizeof(float);
   case DType::FLOAT64:
      return sizeof(double);
   case DType::INT32:
      return sizeof(int32_t);
   case DType::INT64:
      return sizeof(int64_t);
   case DType::BOOL:
      return sizeof(bool);
   }
   throw std::runtime_error("Unknown DType");
};

#endif // DTYPE_HPP
