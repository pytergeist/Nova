#ifndef INODE_H
#define INODE_H

#include <memory>
#include <utility>

#include "AutodiffMeta.hpp"
#include "Node.hpp"

/* TODO: Find a way to remove the static arity from the INode interface/Node,
 * the get_static_output() method here is currently used in Graph<T>, during */

template <typename T> class Graph;

template <typename T> class INode {
 public:
   const std::type_info &in_type() const { return self_->in_type(); }
   const std::type_info &out_type() const { return self_->out_type(); }
   const std::type_info &grad_in_type() const { return self_->grad_in_type(); }
   const std::type_info &grad_out_type() const {
      return self_->grad_out_type();
   }
   std::string_view name() const { return self_->name(); }

   std::vector<ValueID> inputs() const { return inputs_; }
   std::vector<ValueID> outputs() const { return outputs_; }

   bool has_outputs() const { return !outputs_.empty(); };
   bool has_inputs() const { return !inputs_.empty(); };

   std::size_t num_inputs() const { return inputs_.size(); }
   std::size_t num_outputs() const { return outputs_.size(); }

   ValueID get_input(std::size_t idx) const { return inputs_.at(idx); };
   ValueID get_output(std::size_t idx) const { return outputs_.at(idx); };

   template <class Op>
   explicit INode(Op op)
       : self_(std::make_unique<NodeModel<Op>>(std::move(op))){};

   INode(INode &&) noexcept = default;
   INode &operator=(INode &&) noexcept = default;

   INode(const INode &) = delete;
   INode &operator=(const INode &) = delete;

   ~INode() = default;

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
      if (!self_) {
         throw std::runtime_error("INode used after move");
      }
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
   friend Graph<T>;

   std::vector<ValueID> inputs_;
   std::vector<ValueID> outputs_;

   void set_input(std::size_t idx, ValueID vid) {
      resize_inputs(idx + 1);
      inputs_[idx] = vid;
   };

   void set_output(std::size_t idx, ValueID vid) {
      resize_outputs(idx + 1);
      inputs_[idx] = vid;
   };

   void add_input(ValueID vid) { inputs_.push_back(vid); }
   void add_output(ValueID vid) { outputs_.push_back(vid); }

   void reserve_inputs(std::size_t n) { inputs_.reserve(n); }
   void reserve_outputs(std::size_t n) { outputs_.reserve(n); }

   void resize_inputs(std::size_t n) { inputs_.resize(n); }
   void resize_outputs(std::size_t n) { inputs_.resize(n); }

   // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
   struct NodeConcept {
      std::vector<ValueID> inputs;
      std::vector<ValueID> outputs;

      NodeConcept() = default;

      NodeConcept(const NodeConcept &) = delete;
      NodeConcept &operator=(const NodeConcept &) = delete;

      NodeConcept(NodeConcept &&) = delete;
      NodeConcept &operator=(NodeConcept &&) = delete;

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
   // NOLINTEND(misc-non-private-member-variables-in-classes)

   std::unique_ptr<NodeConcept> self_;
};

#endif // INODE_H
