#ifndef TRAITS_H
#define TRAITS_H

#include <cstdint>
#include <initializer_list>
#include <vector>
#include <variant>
#include <any>

// TODO: Create fixed size AutodiffMeta for hot paths

template <typename U> class Tensor;

struct ValueID {
   int32_t idx;
};
struct NodeID {
   int32_t idx;
};

template <typename T> struct AutodiffMeta {
//   using Param = std::variant<int, double, bool>;
   std::vector<Tensor<T>> data; // TODO: migrate this away from std::vector
//   std::unordered_map<std::string, Param> params;
   // NB: in the current impl, params are type erased in meta
   // but must be strongly typed at call site. This means strongtypes
   // must be defined for each ops param type (curr defined in ops/OpParams.h)
   std::any op_param;
   AutodiffMeta() = default;

   explicit AutodiffMeta(std::size_t n) { data.reserve(n); }

   AutodiffMeta(const AutodiffMeta &) = delete;
   AutodiffMeta &operator=(const AutodiffMeta &) = delete;
   AutodiffMeta(AutodiffMeta &&) noexcept = default;
   AutodiffMeta &operator=(AutodiffMeta &&) noexcept = default;

   void emplace_back(const Tensor<T> &y) {data.emplace_back(y); }
   void emplace_back(Tensor<T> &y) {data.emplace_back(y); }

   void push_back(const Tensor<T> &v) { data.emplace_back(v); }
   void push_back(Tensor<T> &&) = delete;

   Tensor<T> &at(std::size_t i) { return data.at(i); }
   const Tensor<T> &at(std::size_t i) const { return data.at(i); }

   bool empty() const { return data.empty(); }
   std::size_t size() const noexcept { return data.size(); }

   Tensor<T> &operator[](std::size_t i) { return data.at(i); }
   const Tensor<T> &operator[](std::size_t i) const { return data.at(i); }

//   template <typename V> void set_param(const std::string &key, V value) {
//      params[key] = value;
//   }

//   template <typename V> V get_param(const std::string &key) const {
//      auto it = params.find(key);
//      FUSION_CHECK(it != params.end(), "Parameter not found");
//      return std::get<V>(it->second);
//   }

   auto begin() { return data.begin(); }
   auto end() { return data.end(); }
   auto begin() const { return data.begin(); }
   auto end() const { return data.end(); }
};

#endif // TRAITS_H
