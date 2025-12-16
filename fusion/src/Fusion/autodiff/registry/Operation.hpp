#ifndef OPERATION_HPP
#define OPERATION_HPP

#include <any>
#include <string>
#include <unordered_map>
#include <vector>

#include "Fusion/common/Checks.hpp"

template <typename U> class ADTensor;

template <typename T, typename Enable = void>
struct static_arity : std::integral_constant<std::size_t, 0> {};


template <typename T> struct Context {
   using CtxValueType = std::variant<ADTensor<T>, int>;
   std::unordered_map<std::string, CtxValueType> saved_result;

   template <typename U> void save(std::string key, U &&data) {
      U fdata = std::forward<U>(data);
      saved_result.insert_or_assign(std::move(key), fdata);
   }

   template <typename U> U &load(std::string &key) {
      auto it = saved_result.find(key);
      if (it == saved_result.end()) {
         FUSION_CHECK(false, "Context::load: key not found: " + key);
      }
      const CtxValueType &v = it->second;
      FUSION_LOG_INFO("Context::load key=", key, " index=", v.index());
      return std::get<U>(v);
   }

   template <typename U> const U &load(const std::string &key) const {
      return std::get<U>(saved_result.at(key));
   }
};

template <typename T, class Op> class Operation {
 public:
   using In = typename Op::In;
   using Out = typename Op::Out;
   using GradIn = typename Op::GradIn;
   using GradOut = typename Op::GradOut;

   Operation() = default;
   explicit Operation(Op op) : op_(std::move(op)) {}

   Out forward(Context<T> &context, In &input) {
      return op_.forward(context, input);
   };
   GradIn backward(Context<T> &context, GradOut &grad_out) {
      return op_.backward(context, grad_out);
   };

   static constexpr std::string_view name = Op::name;

 private:
   Op op_;
};

#endif // OPERATION_HPP
