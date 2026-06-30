#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>

#include "Fusion/TensorFactory.hpp"
#include "Fusion/common/Checks.hpp"

#include "ADTypes.h"
#include "AutodiffMeta.hpp"
#include "Graph.hpp"
#include "Sort.hpp"

template <typename T> class Engine {
 public:
   Engine(GradStore<T> &grad_store) : grad_store_(grad_store) {};

   Engine(const Engine &) = delete;
   Engine &operator=(const Engine &) = delete;

   Engine(Engine &&) = delete;
   Engine &operator=(Engine &&) = delete;

   bool val_buffer_is_empty() const noexcept { return val_buff_.empty(); }

   bool grad_buffer_is_empty() const noexcept { return grad_buff_.empty(); }

   template <class Op>
   std::vector<ValueID> apply_multi(AutodiffMeta<T> &payload,
                                    std::vector<ValueID> &vids) {
      NodeID nid = create_node_and_bind_inputs<Op>(payload, vids);

      INode<T> &node = graph_.get_node(nid);
      AutodiffMeta<T> out = run_forward(node, payload);

      FUSION_CHECK(!out.empty(),
                   "Engine::apply: forward produced empty outputs");

      const bool output_requires_grad = any_input_requires_grad(vids);

      ensure_node_outputs_allocated(nid, out.size(), output_requires_grad);
      write_forward_results(nid, out);

      return node.outputs();
   }

   template <typename Op>
   ValueID apply_single(AutodiffMeta<T> &payload, std::vector<ValueID> &vids) {
      const std::vector<ValueID> out = apply_multi<Op>(payload, vids);
      constexpr std::size_t allowed_outputs = 1;
      if (op_num_outputs_v<typename Op::tag> == allowed_outputs &&
          out.size() == allowed_outputs) {
         return out.front();
      }
      throw std::runtime_error(
          "Engine::apply_single: invalid number of operation outputs produced");
   }

   void backward(const ValueID seed_vid, const bool materialise = true,
                 const bool retain_graph = false) {
      FUSION_CHECK(has_value(seed_vid),
                   "backward() seed ValueID does not refer to a valid value");
      FUSION_CHECK(
          requires_grad_buff_[seed_vid],
          "backward() Invalid call on tensor marked as requires_grad = False");
      prepare_grad_buffers();

      std::vector<NodeID> order = topo_sort_for_backward();
      AutodiffMeta<T> seed = init_seed_grad(seed_vid);

      static_cast<void>(seed);

      for (auto it = order.rbegin(); it != order.rend(); ++it) {
         INode<T> &n = graph_.get_node(NodeID{it->idx});
         FUSION_CHECK(n.has_outputs(), "node has no outputs in backward()");
         AutodiffMeta<T> grad_in;
         for (size_t i = 0; i < n.num_outputs(); ++i) {
            ValueID out_vid = n.get_output(i);
            validate_forward_value_exists(n, out_vid);
            ensure_output_grad_slot(out_vid);
            grad_in.push_back(grad_buff_[out_vid]);
         }

         AutodiffMeta<T> grad_out = safe_apply_backward(n, grad_in);

         FUSION_CHECK(grad_out.size() == n.num_inputs(),
                      "backward arity mismatch");
         accum_input_grads(n, grad_out);
      }

      if (materialise) {
         export_leaf_grads();
      }

      if (retain_graph) {
         throw std::logic_error("retain_graph not implemented");
      }
   }

   void export_leaf_grads() {
      for (auto [vid, binding] : leaf_grad_map_) {
         Tensor<T> &grad = grad_buff_.at(vid);
         grad_store_.set(binding.slot, grad);
      }
   }

   void maybe_mark_leaf(ValueID vid, const bool requires_grad) {
      if (graph_.get_produced_by(vid).nid == -1 && requires_grad) {
         const GradSlotID slot = grad_store_.allocate();
         const LeafGradBinding binding{.vid = vid, .slot = slot};
         leaf_grad_map_.insert({vid, binding});
         requires_grad_buff_[static_cast<std::size_t>(vid)] = requires_grad;
      }
   }

   GradSlotID get_grad_slot(const ValueID vid) const {
      auto it = leaf_grad_map_.find(vid);
      if (it != leaf_grad_map_.end()) {
         const LeafGradBinding binding = it->second;
         FUSION_CHECK(vid == binding.vid,
                      "ValueID out of sync with GradBindings");
         return binding.slot;
      }
      throw std::runtime_error("Gradient not found in persistent grad storage");
   }

   ValueID track_input(const Tensor<T> &raw, const bool requires_grad) {
      const ValueID vid = graph_.new_input_value();
      ensure_value_capacity(vid);
      val_buff_[vid] = raw;
      requires_grad_buff_[vid] = requires_grad;
      maybe_mark_leaf(vid, requires_grad);
      return vid;
   }

   // TODO: replace below
   // Tensor<T> materialise(ValueID vid) {
   //    return val_buff_.at(vid);
   // }
   Tensor<T> materialise(ValueID vid) {
      FUSION_BOUNDS_CHECK(vid, val_buff_.size());
      const Tensor<T> &src = val_buff_[vid];

      std::vector<T> data(src.begin(), src.end());
      return Tensor<T>::from_dense(DenseTensor<T>(src.shape(), std::move(data),
                                                  src.dtype(), src.device()));
   }

   Tensor<T> get_grad(ValueID vid) {
      FUSION_BOUNDS_CHECK(vid, grad_buff_.size());
      return grad_buff_[vid];
   }

   bool has_value(const ValueID vid) const noexcept {
      if (vid < 0) {
         return false;
      }
      const auto idx = static_cast<std::size_t>(vid);
      if (idx >= val_buff_.size()) {
         return false;
      }
      if (!graph_knows(vid)) {
         return false;
      }
      if (!val_buff_[idx].is_initialised()) {
         return false;
      }
      return true;
   }

   void dump_graph(std::ostream &os) const {
      for (size_t i = 0; i < graph_.nodes().size(); ++i) {
         const INode<T> &n = graph_.get_node(NodeID{static_cast<int32_t>(i)});
         os << "Node " << i << " [" << n.name() << "]\n";
      }

      const size_t n = std::min(grad_buff_.size(), graph_.produced_by().size());
      for (size_t i = 0; i < n; ++i) {
         const INode<T> &prod = graph_.get_produced_by(i);
         NodeID nid = prod.nid;
         if (nid >= 0 && static_cast<size_t>(nid) < graph_.nodes().size()) {
            os << "Node idx: " << nid
               << " Node Op: " << graph_.get_node(nid).name() << " ";
            if (val_buff_[i].empty()) {
               os << "[no val]\n";
            } else {
               std::cout << "Node Val: ";
               for (size_t j = 0; j < val_buff_[i].size(); ++j) {
                  os << val_buff_[i][j] << " ";
               }
            }
            if (grad_buff_[i].empty()) {
               os << "[no grad]\n";
            } else {
               std::cout << "Node Grad: ";
               for (size_t j = 0; j < grad_buff_[i].size(); ++j) {
                  os << grad_buff_[i][j] << " ";
               }
               os << "\n";
            }
         }
      }
   }

 private:
   Graph<T> graph_{};
   std::vector<Tensor<T>> val_buff_{};
   std::vector<Tensor<T>> grad_buff_{};
   std::vector<bool> requires_grad_buff_{};
   GradStore<T> &grad_store_{};
   // TODO: make ValueID hashable so it can be used in the below unordered_set
   std::unordered_map<ValueID, LeafGradBinding> leaf_grad_map_;

   void ensure_value_capacity(const ValueID vid) {
      if (val_buff_.size() <= static_cast<size_t>(vid)) {
         val_buff_.resize(static_cast<size_t>(vid) + 1);
      }

      if (requires_grad_buff_.size() <= static_cast<size_t>(vid)) {
         requires_grad_buff_.resize(static_cast<size_t>(vid) + 1);
      }
   }

   bool graph_knows(ValueID vid) const noexcept {
      return static_cast<size_t>(vid) < graph_.produced_by().size();
   }

   bool value_requires_grad(ValueID vid) const noexcept {
      if (vid < 0) {
         return false;
      }

      const auto idx = static_cast<std::size_t>(vid);
      return idx < requires_grad_buff_.size() && requires_grad_buff_[idx];
   }

   bool
   any_input_requires_grad(const std::vector<ValueID> &vids) const noexcept {
      for (const ValueID vid : vids) {
         if (value_requires_grad(vid)) {
            return true;
         }
      }
      return false;
   }

   const Tensor<T> &grad(ValueID vid) const {
      FUSION_BOUNDS_CHECK(vid, grad_buff_.size());
      return grad_buff_[vid];
   }

   void set_grad_buff_size() { grad_buff_.resize(val_buff_.size()); }

   AutodiffMeta<T> run_forward(INode<T> &node, AutodiffMeta<T> &vec) {
      return node.apply_forward(vec);
   }

   template <class Op>
   NodeID create_node_and_bind_inputs(AutodiffMeta<T> &payload,
                                      std::vector<ValueID> &input_vids) {
      FUSION_CHECK(input_vids.size() == payload.size(),
                   "Engine::apply: input_vids size must match payload size");
      NodeID dst = graph_.template build_node<Op>();
      INode<T> &node = graph_.get_node(dst);

      for (size_t i = 0; i < payload.size(); ++i) {
         ValueID vid = input_vids[i];
         graph_.set_node_input(node, vid);
         graph_.append_consumer_table(dst, vid, i);

         const NodeID src = graph_.get_produced_by(vid).nid;
         if (src != -1) {
            graph_.add_edge(src, dst);
         }
      }
      return dst;
   }

   void ensure_node_outputs_allocated(NodeID nid, const std::size_t arity,
                                      const bool output_requires_grad) {
      INode<T> &node = graph_.get_node(nid);
      if (node.has_outputs()) {
         FUSION_CHECK(node.num_outputs() == arity, "node output size mismatch");
         for (size_t i = 0; i < arity; ++i) {
            const ValueID out_vid = node.get_output(i);
            ensure_value_capacity(out_vid);
            requires_grad_buff_[out_vid] = output_requires_grad;
         }
         return;
      }
      for (size_t i = 0; i < arity; ++i) {
         ValueID vid = graph_.new_intermediate_value();
         graph_.set_produced_by(vid, nid, i);
         graph_.set_node_output(node, vid);
         ensure_value_capacity(vid);
         requires_grad_buff_[vid] = output_requires_grad;
      }
   }

   void write_forward_results(NodeID nid, const AutodiffMeta<T> &out) {
      INode<T> &node = graph_.get_node(nid);
      FUSION_BOUNDS_CHECK(0, node.num_outputs());
      FUSION_CHECK(node.num_outputs() == out.size(),
                   "node output size mismatch");

      for (size_t i = 0; i < out.size(); ++i) {
         ValueID vid_i = node.get_output(i);
         ensure_value_capacity(vid_i);
         val_buff_[vid_i] = out[i];
      }
   }

   void prepare_grad_buffers() {
      set_grad_buff_size();
      for (auto &g : grad_buff_) {
         if (g.is_initialised()) {
            g.clear();
         }
      }
   }

   std::vector<NodeID> topo_sort_for_backward() {
      Sort<T> sort_(graph_.nodes().size());
      return sort_.topological_sort(graph_.nodes(), graph_.produced_by(),
                                    graph_.consumed_by(), graph_.node_ids());
   }

   AutodiffMeta<T> init_seed_grad(ValueID vid) {
      grad_buff_[vid] = ones_like(val_buff_[vid]);
      AutodiffMeta<T> v;
      v.push_back(grad_buff_[vid]);
      return v;
   }

   void validate_forward_value_exists(const INode<T> &n,
                                      ValueID out_vid) const {
      FUSION_CHECK(static_cast<size_t>(out_vid) < val_buff_.size(),
                   std::string("val index OOB in backward: ") +
                       std::string(n.name()));
      FUSION_CHECK(val_buff_[out_vid].is_initialised(),
                   std::string("val missing for node output: ") +
                       std::string(n.name()));
   }

   void ensure_output_grad_slot(ValueID out_vid) {
      if (!grad_buff_[out_vid].is_initialised()) {
         grad_buff_[out_vid] = zeros_like(val_buff_[out_vid]);
      }
   }

   AutodiffMeta<T> safe_apply_backward(INode<T> &n, AutodiffMeta<T> &gin) {
      try {
         return n.apply_backward(gin);
      } catch (const std::exception &e) {
         throw std::runtime_error(std::string("apply_backward threw in op ") +
                                  std::string(n.name()) + ": " +
                                  std::string(e.what()));
      }
   }

   void accum_input_grads(const INode<T> &n, const AutodiffMeta<T> &gout) {
      for (size_t j = 0; j < n.num_inputs(); ++j) {
         const ValueID in_vid = n.get_input(j);
         Tensor<T> &dst = grad_buff_[in_vid];
         const Tensor<T> &src = gout[j];

         if (!dst.is_initialised()) {
            dst = src;
         } else {
            dst = dst + src;
         }
      }
   }
};

#endif // ENGINE_HPP
