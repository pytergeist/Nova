#ifndef INODE_H
#define INODE_H
#include <memory>
#include <any>
#include <utility>
#include "Node.h"
#include "Traits.h"

class INode {
  public:
    const std::type_info& in_type()  const { return self_->in_type(); }
    const std::type_info& out_type() const { return self_->out_type(); }
    const std::type_info& grad_in_type()  const { return self_->grad_in_type(); }
    const std::type_info& grad_out_type() const { return self_->grad_out_type(); }

    std::vector<ValueID> inputs;
    std::vector<ValueID> outputs;

    template<class Op>
    explicit INode(Op op) : self_(std::make_unique<NodeModel<Op>>(std::move(op))) {};

    INode(INode&&) noexcept = default;
    INode& operator=(INode&&) noexcept = default;
    INode(const INode&) = delete;
    INode& operator=(const INode&) = delete;

    std::any forward(const std::any& input) {return self_->forward(input);};
    std::any backward(std::any& grad_out) {return self_->backward(grad_out);};

    std::any apply_forward(const std::any& input) {
      if (input.type() != in_type())
        throw std::runtime_error("type mismatch");
      std::any out = self_->forward(input);
      if (out.type() != out_type())
        throw std::runtime_error("bad out type");
      return out;
  }

  std::any apply_backward(const std::any& grad_out) {
    if (!self_) throw std::runtime_error("INode used after move");
    if (grad_out.type() != grad_out_type())
      throw std::runtime_error(std::string("grad type mismatch: expected ")
                               + grad_out_type().name() + " got " + grad_out.type().name());
    std::any gin = self_->backward(const_cast<std::any&>(grad_out));
    if (gin.type() != grad_in_type())
      throw std::runtime_error(std::string("bad grad_in type: expected ")
                               + grad_in_type().name() + " got " + gin.type().name());
    return gin;
  }


    template<class ConcreteOp>
    typename ConcreteOp::Out forward_t(const typename ConcreteOp::In& input) {
      std::any out_any = forward(std::any{input});
      return std::any_cast<typename ConcreteOp::Out>(out_any);
    };

    template<class ConcreteOp>
    typename ConcreteOp::GradIn backward_t(typename ConcreteOp::GradOut& grad_out) {
      std::any grad_out_any = grad_out;
      std::any grad_in_any  = backward(grad_out_any);
      return std::any_cast<typename ConcreteOp::GradIn>(std::move(grad_in_any));
    }

    std::uint16_t get_static_num_outputs() {return self_->get_static_num_outputs();};
    std::uint16_t get_static_num_inputs() {return self_->get_static_num_inputs();};


    private:
      struct NodeConcept {
        std::vector<ValueID> inputs;
    	std::vector<ValueID> outputs;

        virtual ~NodeConcept() = default;
        virtual std::any forward(const std::any& input) = 0;
        virtual std::any backward(std::any& grad_out) = 0;

        virtual const std::type_info& in_type() = 0;
        virtual const std::type_info& out_type() = 0;
        virtual const std::type_info& grad_in_type() = 0;
        virtual const std::type_info& grad_out_type() = 0;

        virtual uint16_t get_static_num_outputs() = 0;
        virtual uint16_t get_static_num_inputs() = 0;
      };
      template<class Op>
      struct NodeModel : NodeConcept {

        using In      = typename Op::In;
        using Out     = typename Op::Out;
        using GradIn  = typename Op::GradIn;
        using GradOut = typename Op::GradOut;

        std::vector<ValueID> inputs;
    	std::vector<ValueID> outputs;

        explicit NodeModel(Op op) : node_(std::move(op)) {}

        std::any forward(const std::any& input) {
        const In& x = std::any_cast<const In&>(input);
        auto y = node_.run_forward(x);
        return std::any{std::move(y)};
        };

        std::any backward(std::any& grad_out) {
          auto& grad_out_cast = std::any_cast<GradOut&>(grad_out);
          auto grad_in = node_.run_backward(grad_out_cast);
          return std::any{std::move(grad_in)};
        }

        std::uint16_t get_static_num_outputs() {
          return node_.KStaticNumOutputs;
        }

        std::uint16_t get_static_num_inputs() {
          return node_.KStaticNumInputs;
        }

        const std::type_info& in_type() {return typeid(In);};
        const std::type_info& out_type() {return typeid(Out);};
        const std::type_info& grad_in_type() {return typeid(GradIn);};
        const std::type_info& grad_out_type() {return typeid(GradOut);};

        Node<Op> node_;

      };
  std::unique_ptr<NodeConcept> self_;
  };



#endif // INODE_H
