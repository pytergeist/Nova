// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <unordered_set>

#include "Fusion/TensorFactory.hpp"
#include "Fusion/common/Checks.hpp"

#include "ADTypes.h"
#include "AutodiffMeta.hpp"
#include "BackwardResult.hpp"
#include "Graph.hpp"
#include "Sort.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
inline void hash_combine(std::size_t &seed, std::size_t v) noexcept {
   seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)

struct ShapeHash {
   std::size_t
   operator()(const std::vector<std::size_t> &shape) const noexcept {
      std::size_t seed = 0; // NOLINT(misc-const-correctness)
      for (auto s : shape)
         hash_combine(seed, s);
      return seed;
   }
};

template <typename T> class Engine {
 public:
   Engine() = default;

   Engine(const Engine &) = delete;
   Engine &operator=(const Engine &) = delete;

   Engine(Engine &&) = delete;
   Engine &operator=(Engine &&) = delete;

   ~Engine() = default;

   template <class Op>
   ValueID apply(AutodiffMeta<T> &payload, std::vector<ValueID> &vids) {
      NodeID nid = create_node_and_bind_inputs<Op>(payload, vids);

      INode<T> &node = graph_.get_node(nid);
      AutodiffMeta<T> out = run_forward(node, payload);

      FUSION_CHECK(!out.empty(),
                   "Engine::apply: forward produced empty outputs");

      ensure_node_outputs_allocated(nid, out.size());
      write_forward_results(nid, out);

      // TODO: need to eventually return all outputs? (vector<ValueID>)
      return node.get_output(0);
   }

   BackwardResult<T> backward(ValueID seed_vid, bool materialise = true,
                              bool retain_graph = false) {
      prepare_grad_buffers();

      std::vector<NodeID> order = topo_sort_for_backward();
      AutodiffMeta<T> seed = init_seed_grad(seed_vid);

      static_cast<void>(seed);

      for (auto it = order.rbegin(); it != order.rend(); ++it) {
         INode<T> &n = graph_.get_node(NodeID{it->idx});
         FUSION_CHECK(n.has_outputs(), "node has no outputs in backward()");

         const ValueID out_vid = n.get_output(0);
         validate_forward_value_exists(n, out_vid);
         ensure_output_grad_slot(out_vid);

         AutodiffMeta<T> grad_in;
         grad_in.push_back(grad_buff_[out_vid]);
         AutodiffMeta<T> grad_out = safe_apply_backward(n, grad_in);

         FUSION_CHECK(grad_out.size() == n.num_inputs(),
                      "backward arity mismatch");
         accum_input_grads(n, grad_out);
      }

      BackwardResult<T> result;

      if (materialise) {
         result = materialise_leaf_grads();
         return result;
      }

      if (retain_graph) {
         throw std::logic_error("retain_graph not implemented");
      }
      return result;
   }

   void maybe_mark_leaf(ValueID vid, const bool requires_grad) {
      if (graph_.get_produced_by(vid).nid == -1 && requires_grad) {
         requires_grad_set_.insert(vid);
      }
   }

   BackwardResult<T> materialise_leaf_grads() {
      BackwardResult<T> result;
      for (std::int64_t vid : requires_grad_set_) {
         result.grads.try_emplace(vid, grad_buff_[vid]);
      }
      FUSION_CHECK(!result.empty(),
                   "backward result is empty - no gradients to attatch");
      return result;
   }

   ValueID track_input(const RawTensor<T> &raw, const bool requires_grad) {
      const ValueID vid = graph_.new_input_value();
      ensure_value_capacity(vid);
      val_buff_[vid] = raw;
      maybe_mark_leaf(vid, requires_grad);
      return vid;
   }

   RawTensor<T> materialise(ValueID vid) {
      const RawTensor<T> &src = val_buff_[vid];

      std::vector<T> data(src.begin(), src.end());
      return RawTensor<T>(src.shape(), std::move(data), src.dtype(),
                          src.device());
   }

   RawTensor<T> get_grad(ValueID vid) {
      FUSION_BOUNDS_CHECK(vid, val_buff_.size());
      return grad_buff_[vid];
   }

   bool has_value(ValueID vid) const noexcept {
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
   std::vector<RawTensor<T>> val_buff_{};
   std::vector<RawTensor<T>> grad_buff_{};
   // TODO: make ValueID hashable so it can be used in the below unordered_set
   std::unordered_set<std::int64_t> requires_grad_set_{};

   void ensure_value_capacity(ValueID vid) {
      if (val_buff_.size() <= static_cast<size_t>(vid)) {
         val_buff_.resize(static_cast<size_t>(vid) + 1);
      }
   }

   bool graph_knows(ValueID vid) const noexcept {
      return static_cast<size_t>(vid) < graph_.produced_by().size();
   }

   const RawTensor<T> &grad(ValueID vid) const {
      FUSION_BOUNDS_CHECK(vid, grad_buff_.size());
      return grad_buff_[vid];
   }

   template <class Op> ValueID feed(AutodiffMeta<T> v) {
      NodeID dst_nid = graph_.template build_node<Op>(v);
      INode<T> &node = graph_.get_node(dst_nid);
      // Arbitrily set to 0 for single tensor feed
      // TODO: make return type vec<ValueID> to allow for multi output??? Not
      // sure this is needed
      ValueID vid = node.outputs.at(0);
      write_val(vid, v[0]);
      return vid;
   }

   void set_grad_buff_size() { grad_buff_.resize(val_buff_.size()); }

   AutodiffMeta<T> run_forward(INode<T> &node, AutodiffMeta<T> &vec) {
      return node.apply_forward(vec);
   }

   template <class Op>
   NodeID create_node_and_bind_inputs(AutodiffMeta<T> &payload,
                                      std::vector<ValueID> &input_vids) {
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

   void ensure_node_outputs_allocated(NodeID nid, size_t arity) {
      INode<T> &node = graph_.get_node(nid);
      if (node.has_outputs()) {
         FUSION_CHECK(node.num_outputs() == arity, "node output size mismatch");
         return;
      }
      for (size_t i = 0; i < arity; ++i) {
         ValueID vid = graph_.new_intermediate_value();
         graph_.set_produced_by(vid, nid, i);
         graph_.set_node_output(node, vid);
         ensure_value_capacity(vid);
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
         if (g.is_initialised())
            g.clear();
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
         RawTensor<T> &dst = grad_buff_[in_vid];
         const RawTensor<T> &src = gout[j];

         if (!dst.is_initialised()) {
            dst = src;
         } else {
            dst = dst + src;
         }
      }
   }
};

#endif // ENGINE_HPP
