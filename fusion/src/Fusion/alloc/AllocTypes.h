// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef ALLOC_TYPES_H
#define ALLOC_TYPES_H

#include <cstddef>

struct Alignment {
   std::size_t value;
   operator std::size_t() const noexcept { return value; }
};

#endif // ALLOC_TYPES_H
