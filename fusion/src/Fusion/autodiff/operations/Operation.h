#ifndef OPERATION_H
#define OPERATION_H

#include <string>

template <typename T>
struct BinaryType {std::vector<T> a; std::vector<T> b;};

template <typename T>
struct UnaryType {std::vector<T> a;};

template <typename T>
struct Context {
   std::unordered_map<std::string, BinaryType<T>> saved_result;
   void save(std::string key, const BinaryType<T>& bt) {
     saved_result[key] = bt;
   }

   void save(std::string key, const UnaryType<T>& ut) {
     saved_result[key] = ut;
   }

   BinaryType<T> load(std::string key) {
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

    Out forward(Context<T>& context, const In& input) {return op_.forward(context, input);};
    GradIn backward(Context<T>& context, GradOut& grad_out) {return op_.backward(context, grad_out);};

    static constexpr std::string_view name() {return Op::name();};

	private:
          Op op_;


};

#endif // OPERATION_H
