// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef LAYOUT_HPP
#define LAYOUT_HPP

#include <vector>

// TODO: evaluate this impl
inline bool calc_contiguous(const std::vector<std::size_t> &shape,
                            const std::vector<std::size_t> &strides) noexcept {
   if (shape.empty()) {
      return true;
   }
   std::size_t expected = 1;
   for (std::size_t i = 0; i < shape.size(); ++i) {
      if (strides[i] != expected) {
         return false;
      }
      expected *= shape[(i + 1 < shape.size()) ? i + 1 : i];
      if (i + 1 == shape.size())
         expected = 1;
   }
   return true;
}

#endif // LAYOUT_HPP
