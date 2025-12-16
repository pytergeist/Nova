#ifndef NODE_H
#define NODE_H

#include <memory>

#include "AutodiffMeta.hpp"
#include "registry/Operation.hpp"

template <typename T, class Op> class Node {
 public:
   using In = typename Op::In;
   using Out = typename Op::Out;
   using GradIn = typename Op::GradIn;
   using GradOut = typename Op::GradOut;

   Node() = default;
   Node(Op op) : op_(std::move(op)) {};

   using CleanOut = std::remove_cvref_t<Out>;
   using CleanIn = std::remove_cvref_t<In>;
   static constexpr std::size_t KStaticNumOutputs =
       static_arity<CleanOut>::value;
   static constexpr std::size_t KStaticNumInputs = static_arity<CleanIn>::value;

   void set_inputs(In inputs) { inputs_ = std::move(inputs); };

   Out run_forward(In &input) {

      output_ = op_.forward(ctx_, input);
      fwd_done_ = true;
      return std::move(output_);
   };

   GradIn run_backward(GradOut &grad_out) {
      grad_input_ = op_.backward(ctx_, grad_out);
      bwd_done_ = true;
      return std::move(grad_input_);
   }

 private:
   Op op_{};
   Context<T> ctx_{};
   In inputs_{};
   Out output_{};
   GradIn grad_input_{};
   bool fwd_done_{false};
   bool bwd_done_{false};
};

#endif // NODE_H
