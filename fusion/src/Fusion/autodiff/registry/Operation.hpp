#ifndef OPERATION_HPP
#define OPERATION_HPP

#include <any>
#include <string>
#include <unordered_map>
#include <vector>

#include "Fusion/common/Checks.hpp"
#include "Fusion/core/tensor/Tensor.hpp"

template <typename T> struct Context {
   using CtxValueType = std::variant<Tensor<T>, int>;
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
   using tag = Op::tag;
   using In = Op::In;
   using Out = Op::Out;
   using GradIn = Op::GradIn;
   using GradOut = Op::GradOut;

   static constexpr std::string_view name = OpTraits<tag>::name;
   static constexpr OpSchema schema = OpTraits<tag>::schema;

   Operation() = default;
   explicit Operation(Op op) : op_(std::move(op)) {}

   static std::size_t input_arity() {
      if (op_has_fixed_inputs_v<tag>) {
         return op_inputs_v<tag>.arity;
      }
      throw std::runtime_error("Variadic input arity: Not implemented");
   }

   static std::size_t output_arity() {
      if (op_has_fixed_outputs_v<tag>) {
         return op_outputs_v<tag>.arity;
      }
      throw std::runtime_error("Variadic output arity: Not implemented");
   }

   Out forward(Context<T> &context, In &input) {
      return op_.forward(context, input);
   };
   GradIn backward(Context<T> &context, GradOut &grad_out) {
      return op_.backward(context, grad_out);
   };

 private:
   Op op_;
};

#endif // OPERATION_HPP
