// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef AUTODIFF_META_HPP
#define AUTODIFF_META_HPP

#include <any>
#include <cstdint>
#include <initializer_list>
#include <variant>
#include <vector>

#include "ADTypes.h"

// TODO: Create fixed size AutodiffMeta for hot paths

template <typename U> class RawTensor;

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)
template <typename T> struct AutodiffMeta {
   std::vector<RawTensor<T>> data;
   // NB: in the current impl, params are type erased in meta
   // but must be strongly typed at call site. This means strongtypes
   // must be defined for each ops param type (curr defs in ops/OpParams.h)
   std::any op_param;

   AutodiffMeta() = default;
   explicit AutodiffMeta(std::size_t n) { data.reserve(n); }

   AutodiffMeta(const AutodiffMeta &) = delete;
   AutodiffMeta &operator=(const AutodiffMeta &) = delete;

   AutodiffMeta(AutodiffMeta &&) noexcept = default;
   AutodiffMeta &operator=(AutodiffMeta &&) noexcept = default;

   ~AutodiffMeta() = default;

   void emplace_back(const RawTensor<T> &y) { data.emplace_back(y); }
   void emplace_back(RawTensor<T> &y) { data.emplace_back(y); }

   void push_back(const RawTensor<T> &v) { data.emplace_back(v); }
   void push_back(RawTensor<T> &&) = delete;

   RawTensor<T> &at(std::size_t i) { return data.at(i); }
   const RawTensor<T> &at(std::size_t i) const { return data.at(i); }

   bool empty() const { return data.empty(); }
   std::size_t size() const noexcept { return data.size(); }

   RawTensor<T> &operator[](std::size_t i) { return data.at(i); }
   const RawTensor<T> &operator[](std::size_t i) const { return data.at(i); }

   auto begin() { return data.begin(); }
   auto end() { return data.end(); }
   auto begin() const { return data.begin(); }
   auto end() const { return data.end(); }
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

#endif // AUTODIFF_META_HPP
