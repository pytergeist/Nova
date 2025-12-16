#ifndef TRAITS_H
#define TRAITS_H

#include <any>
#include <cstdint>
#include <initializer_list>
#include <variant>
#include <vector>

#include "ADTypes.hpp"

// TODO: Create fixed size AutodiffMeta for hot paths

template <typename U> class ADTensor;

// NOLINTBEGIN(misc-non-private-member-variables-in-classes)
template <typename T> struct AutodiffMeta {
   std::vector<ADTensor<T>> data;
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

   void emplace_back(const ADTensor<T> &y) { data.emplace_back(y); }
   void emplace_back(ADTensor<T> &y) { data.emplace_back(y); }

   void push_back(const ADTensor<T> &v) { data.emplace_back(v); }
   void push_back(ADTensor<T> &&) = delete;

   ADTensor<T> &at(std::size_t i) { return data.at(i); }
   const ADTensor<T> &at(std::size_t i) const { return data.at(i); }

   bool empty() const { return data.empty(); }
   std::size_t size() const noexcept { return data.size(); }

   ADTensor<T> &operator[](std::size_t i) { return data.at(i); }
   const ADTensor<T> &operator[](std::size_t i) const { return data.at(i); }

   auto begin() { return data.begin(); }
   auto end() { return data.end(); }
   auto begin() const { return data.begin(); }
   auto end() const { return data.end(); }
};
// NOLINTEND(misc-non-private-member-variables-in-classes)

#endif // TRAITS_H
