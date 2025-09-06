#ifndef OPERATION_H
#define OPERATION_H

#include <string>
#include <any>

template <typename T>
struct BinaryType {std::vector<T> a; std::vector<T> b;};

template <typename T>
struct UnaryType {std::vector<T> a;};

struct Context {
   std::unordered_map<std::string, std::any> saved_result;

   template <typename U>
   void save(std::string key, U&& data) {
     saved_result[std::move(key)] = std::any(std::forward<U>(data));
   }

   template <typename U>
   U& load(std::string& key) {
     return std::any_cast<U&>(saved_result.at(key));
   }

   template <typename U>
    const U& load(const std::string& key) const {
        return std::any_cast<const U&>(saved_result.at(key));
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

    Out forward(Context& context, const In& input) {return op_.forward(context, input);};
    GradIn backward(Context& context, GradOut& grad_out) {return op_.backward(context, grad_out);};

    static constexpr std::string_view name() {return Op::name();};

	private:
          Op op_;


};

#endif // OPERATION_H
