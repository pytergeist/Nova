#ifndef INODE_H
#define INODE_H
#include <memory>

class INode {
  public:
    template<class Op>
    explicit Node(Op& op) : self_(std::make_unique<NodeImpl<Op>>(std::move(op))) {}

    std::any In forward(const std::any& input) = {return self_->forward(input);};
    std::any GradIn backward(std::any& grad_out) = {return self_->backward(grad_out);};

    }

    private:
      struct NodeConcept {
        virtual ~INodeImpl() = default;
        virtual In forward(const std::any& input) = 0;
        virtual GradIn backward(std::any& grad_out) = 0;
      }
      template<class Op>
      struct NodeModel : NodeConcept {

        explicit NodeModel(Op& op) : Node_(std::move(op)) {}

        std::any run_forward(const std::any& input) {
        using In = typename Op::In;
        const In& x = std::any_cast<const In&>(input);
        auto y = node_.run_forward(x);
        return std::any{std::move(y)};
        };

        std::any run_backward(std::any& grad_out) {
          using GradOut = typename Op::GradOut;
          auto& grad_out = std::any_cast<GradOut&>(grad_out);
          auto grad_in = node_.run_backward(grad_out);
          return std::any{std::move(grad_in)};

        }

        Node<Op> node_;

      }
  std::unique_ptr<Concept> self_;
  }



#endif // INODE_H
