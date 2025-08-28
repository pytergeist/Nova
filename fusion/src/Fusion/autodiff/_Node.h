#ifndef _NODE_H
#define _NODE_H

template <typename T, class Op> class Node {
  Tensor<T> value_;
  Op operation_;
  std::tuple<Node<T>...> parents_;
  std::bool requiresGrad;
  std::string role_;
  T grad_;

public:
  explicit Node() = default;
  ~Node() = default;

  void update_node_gradient(Tensor<T> grad_output) {
    if (grad_output.size() == 0) {
      return;
    }
    expected_shape = this->value_.get_shape();
    if (expected_shape != expected_shape) {
      if (grad_output.ndim() == 2) {
        Tensor<T> grad_output = grad_output.diag();
      }
    }
  }
  if (this->Grad_.size() == 0) {
    this->Grad_ = grad_output;
  } else {
    this->Grad_ += grad_output;
  }
};

#endif // _NODE_H
