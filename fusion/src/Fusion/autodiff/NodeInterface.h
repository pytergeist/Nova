#ifndef INODE_H
#define INODE_H
#include <memory>
#include <any>
#include <utility>
#include "Node.h"

class INode {
  public:
    template<class Op>
    explicit INode(Op& op) : self_(std::make_unique<NodeModel<Op>>(op)) {};

    std::any forward(const std::any& input) {return self_->forward(input);};
    std::any backward(std::any& grad_out) {return self_->backward(grad_out);};

    private:
      struct NodeConcept {
        virtual ~NodeConcept() = default;
        virtual std::any forward(const std::any& input) = 0;
        virtual std::any backward(std::any& grad_out) = 0;
      };
      template<class Op>
      struct NodeModel : NodeConcept {

        explicit NodeModel(Op& op) : node_(std::move(op)) {}

        std::any forward(const std::any& input) {
        using In = typename Op::In;
        const In& x = std::any_cast<const In&>(input);
        auto y = node_.run_forward(x);
        return std::any{std::move(y)};
        };

        std::any backward(std::any& grad_out) {
          using GradOut = typename Op::GradOut;
          auto& grad_out_cast = std::any_cast<GradOut&>(grad_out);
          auto grad_in = node_.run_backward(grad_out_cast);
          return std::any{std::move(grad_in)};
        }

        Node<Op> node_;

      };
  std::unique_ptr<NodeConcept> self_;
  };



#endif // INODE_H
