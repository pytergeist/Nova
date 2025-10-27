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
   const std::type_info &grad_out_type() const {
      return self_->grad_out_type();
   }
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

   AutodiffMeta<T> forward(AutodiffMeta<T> &input) {
      return self_->forward(input);
   };
   AutodiffMeta<T> backward(AutodiffMeta<T> &grad_out) {
      return self_->backward(grad_out);
   };

   AutodiffMeta<T> apply_forward(AutodiffMeta<T> &input) {
      AutodiffMeta<T> out = self_->forward(input);
      return out;
   }

   AutodiffMeta<T> apply_backward(AutodiffMeta<T> &grad_out) {
      if (!self_)
         throw std::runtime_error("INode used after move");
      AutodiffMeta<T> gin = self_->backward(grad_out);
      return gin;
   }

   std::size_t get_static_num_outputs() {
      return self_->get_static_num_outputs();
   };
   std::size_t get_static_num_inputs() {
      return self_->get_static_num_inputs();
   };

 private:
   struct NodeConcept {
      std::vector<ValueID> inputs;
      std::vector<ValueID> outputs;

      virtual ~NodeConcept() = default;
      virtual AutodiffMeta<T> forward(AutodiffMeta<T> &input) = 0;
      virtual AutodiffMeta<T> backward(AutodiffMeta<T> &grad_out) = 0;

      virtual const std::type_info &in_type() const = 0;
      virtual const std::type_info &out_type() const = 0;
      virtual const std::type_info &grad_in_type() const = 0;
      virtual const std::type_info &grad_out_type() const = 0;
      virtual std::string_view name() const = 0;

      virtual size_t get_static_num_outputs() const = 0;
      virtual size_t get_static_num_inputs() const = 0;
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

      AutodiffMeta<T> forward(AutodiffMeta<T> &input) override {
         auto y = node_.run_forward(input);
         return y;
      };

      AutodiffMeta<T> backward(AutodiffMeta<T> &grad_out) override {
         AutodiffMeta<T> grad_in = node_.run_backward(grad_out);
         return grad_in;
      }

      std::size_t get_static_num_outputs() const override {
         return node_.KStaticNumOutputs;
      }

      std::size_t get_static_num_inputs() const override {
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

      Node<T, Op> node_;
   };
   std::unique_ptr<NodeConcept> self_;
};

#endif // INODE_H
