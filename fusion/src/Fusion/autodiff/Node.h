#ifndef _NODE_H
#define _NODE_H

#include <memory>
#include "../ops/Operation.h"


template <class Op>
class Node {
  public:
    using In = typename Op::In;
    using Out = typename Op::Out;
    using GradIn = typename Op::GradIn;
    using GradOut = typename Op::GradOut;

  Node() = default;
  Node(Op op) : op_(std::move(op)) {};

  void set_inputs(In inputs) { inputs_ = std::move(inputs);};

  Out run_forward(const In& input) {
    output_ = op_.forward(ctx_, input);
    fwd_done_ = true;
    return output_;
  };

  GradIn run_backward(GradOut& grad_out) {
    grad_input_ = op_.backward(ctx_, grad_out);
    bwd_done_ = true;
    return grad_input_;
  }


  private:
    Op op_{};
    Context ctx_{};
    In inputs_{};
    Out output_{};
    GradIn grad_input_{};
    bool fwd_done_{false};
    bool bwd_done_{false};



};

#endif // _NODE_H
