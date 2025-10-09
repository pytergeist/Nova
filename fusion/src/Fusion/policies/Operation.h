#ifndef OPERATION_H
#define OPERATION_H

#include <vector>
#include <string>
#include <any>
#include <unordered_map>
#include "../Tensor.h"

template <typename T>
struct BinaryType {std::vector<T> a; std::vector<T> b;};

template <typename T>
struct UnaryType {std::vector<T> a;};

template <typename T, typename Enable = void>
struct static_arity : std::integral_constant<std::uint16_t, 0> {};


template <typename T>
struct static_arity<UnaryType<T>, void>
    : std::integral_constant<std::uint16_t, 1> {};

template <typename T>
struct static_arity<BinaryType<T>, void>
    : std::integral_constant<std::uint16_t, 2> {};

template <typename T>
struct Context {
   std::unordered_map<std::string, Tensor<T>> saved_result;

   template <typename U>
   void save(std::string key, U&& data) {
     saved_result.insert_or_assign(std::move(key), std::move(data));
   }

   template <typename U>
   U& load(std::string& key) {
     return saved_result.at(key);
   }

   template <typename U>
    const U& load(const std::string& key) const {
        return saved_result.at(key);
    }
};

template<typename T, class Op>
class Operation {
  public:
 	using In = typename Op::In;
 	using Out = typename Op::Out;
 	using GradIn = typename Op::GradIn;
    using GradOut = typename Op::GradOut;

    Operation() = default;
    explicit Operation(Op op) : op_(std::move(op)) {}

    Out forward(Context<T>& context, In& input) {return op_.forward(context, input);};
    GradIn backward(Context<T>& context, GradOut& grad_out) {return op_.backward(context, grad_out);};

    inline static constexpr std::string_view name = Op::name;

	private:
          Op op_;
};

#endif // OPERATION_H
