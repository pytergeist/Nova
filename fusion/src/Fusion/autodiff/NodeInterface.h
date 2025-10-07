#ifndef INODE_H
#define INODE_H

#include "Node.h"
#include "Traits.h"
#include <memory>
#include <utility>

template <typename T> class INode {
public:
  const std::type_info &in_type() const { return self_->in_type(); }
  const std::type_info &out_type() const { return self_->out_type(); }
  const std::type_info &grad_in_type() const { return self_->grad_in_type(); }
  const std::type_info &grad_out_type() const { return self_->grad_out_type(); }
  std::string_view name() const { return self_->name(); }

  std::vector<ValueID> inputs;
  std::vector<ValueID> outputs;

  template <class Op>
  explicit INode(Op op)
      : self_(std::make_unique<NodeModel<Op>>(std::move(op))){};

  INode(INode &&) noexcept = default;
  INode &operator=(INode &&) noexcept = default;
  INode(const INode &) = delete;
  INode &operator=(const INode &) = delete;

  MultiTensor<T> forward(const MultiTensor<T> &input) {
    return self_->forward(input);
  };
  MultiTensor<T> backward(MultiTensor<T> &grad_out) {
    return self_->backward(grad_out);
  };

  MultiTensor<T> apply_forward(const MultiTensor<T> &input) {
    MultiTensor<T> out = self_->forward(input);
    return out;
  }

  MultiTensor<T> apply_backward(MultiTensor<T> &grad_out) {
    if (!self_)
      throw std::runtime_error("INode used after move");
    MultiTensor<T> gin = self_->backward(grad_out);
    return gin;
  }

  std::uint16_t get_static_num_outputs() {
    return self_->get_static_num_outputs();
  };
  std::uint16_t get_static_num_inputs() {
    return self_->get_static_num_inputs();
  };

private:
  struct NodeConcept {
    std::vector<ValueID> inputs;
    std::vector<ValueID> outputs;

    virtual ~NodeConcept() = default;
    virtual MultiTensor<T> forward(const MultiTensor<T> &input) = 0;
    virtual MultiTensor<T> backward(MultiTensor<T> &grad_out) = 0;

    virtual const std::type_info &in_type() const = 0;
    virtual const std::type_info &out_type() const = 0;
    virtual const std::type_info &grad_in_type() const = 0;
    virtual const std::type_info &grad_out_type() const = 0;
    virtual std::string_view name() const = 0;

    virtual uint16_t get_static_num_outputs() const = 0;
    virtual uint16_t get_static_num_inputs() const = 0;
  };
  template <class Op> struct NodeModel : NodeConcept {

    using In = typename Op::In;
    using Out = typename Op::Out;
    using GradIn = typename Op::GradIn;
    using GradOut = typename Op::GradOut;

    std::vector<ValueID> inputs;
    std::vector<ValueID> outputs;

    explicit NodeModel(Op op) : node_(std::move(op)) {}

    std::string_view name() const override { return Op::name; }

    MultiTensor<T> forward(const MultiTensor<T> &input) override {
      auto y = node_.run_forward(input);
      return y;
    };

    MultiTensor<T> backward(MultiTensor<T> &grad_out) override {
      MultiTensor<T> grad_in = node_.run_backward(grad_out);
      return grad_in;
    }

    std::uint16_t get_static_num_outputs() const override {
      return node_.KStaticNumOutputs;
    }

    std::uint16_t get_static_num_inputs() const override {
      return node_.KStaticNumInputs;
    }

    const std::type_info &in_type() const override { return typeid(In); };
    const std::type_info &out_type() const override { return typeid(Out); };
    const std::type_info &grad_in_type() const override {
      return typeid(GradIn);
    };
    const std::type_info &grad_out_type() const override {
      return typeid(GradOut);
    };

    Node<Op> node_;
  };
  std::unique_ptr<NodeConcept> self_;
};

#endif // INODE_H
