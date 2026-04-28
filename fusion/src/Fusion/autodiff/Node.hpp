#ifndef NODE_HPP
#define NODE_HPP

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

   static constexpr std::string_view KName = Op::name;
   static constexpr OpSchema KSchema = Op::schema;

   void set_inputs(In inputs) { inputs_ = std::move(inputs); };

   Out run_forward(In &input) {
      fwd_done_ = true;
      return op_.forward(ctx_, input);
   }

   GradIn run_backward(GradOut &grad_out) {
      bwd_done_ = true;
      return op_.backward(ctx_, grad_out);
   }

 private:
   Op op_{};
   Context<T> ctx_{};
   In inputs_{};
   bool fwd_done_{false};
   bool bwd_done_{false};
};

#endif // NODE_HPP
