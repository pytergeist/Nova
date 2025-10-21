#ifndef ENGINE_H
#define ENGINE_H

#include <memory>
#include <ostream>
#include <iostream>

#include "../TensorFactory.h"
#include "../common/Checks.h"
#include "Graph.h"
#include "Sort.h"
#include "Traits.h"

// TODO: general TODO, move private members of all autodiff classes into private

template <typename T> class Engine {
public:
  Engine() : graph_{}, val_buff_{}, grad_buff_{} {};

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&) = delete;
  Engine& operator=(Engine&&) = delete;

  template <class Op> ValueID apply(AutodiffMeta<T> &&payload) {
    size_t num = payload.size();
    std::vector<ValueID> vids;
    vids.reserve(num);
    for (size_t i = 0; i < num; i++) {
      vids.push_back(feed_raw(payload[i]));
    }
    return apply<Op>(vids);
  }

  void backward(ValueID seed_vid) {
    set_grad_buff_size();
    for (auto &g : grad_buff_) {
      if (g.is_initialised())
        g.clear();
    };
    Sort<T> sort_(graph_.nodes.size());
    std::vector<NodeID> sorted_nodes = sort_.topological_sort(
        graph_.nodes, graph_.produced_by, graph_.consumed_by, graph_.node_ids);
    AutodiffMeta<T> gradVec =
        grad_init(seed_vid, 0); // TODO: This chooses a single sink for the seed vid for now
    for (auto it = sorted_nodes.rbegin(); it != sorted_nodes.rend(); ++it) {
      auto &n = graph_.nodes[it->idx];
      FUSION_CHECK(!n.outputs.empty(), "node has no outputs in backward()");
      auto &inputs = n.inputs;
      auto output_id = n.outputs[0];

      FUSION_CHECK(static_cast<size_t>(output_id.idx) < val_buff_.size(),
                   "output_id out of range for val_buff_ (op=" +
                       std::string(n.name()) + ")");
      FUSION_CHECK(val_buff_[output_id.idx].is_initialised(),
                   "val_buff_[output_id] not initialised for node output (op=" +
                       std::string(n.name()) + ")");

      if (!grad_buff_[output_id.idx].is_initialised()) {
        grad_buff_[output_id.idx] = zeros_like(val_buff_[output_id.idx]);
      }

      std::cerr << "[backward] op=" << n.name() << " inputs=" << inputs.size()
                << " out_vid=" << output_id.idx << " out_shape=(";
      {
        const auto &sh = val_buff_[output_id.idx].storage->shape();
        for (size_t k = 0; k < sh.size(); ++k)
          std::cerr << sh[k] << (k + 1 < sh.size() ? "," : "");
        std::cerr << ")\n";
      }

      AutodiffMeta<T> grad_in;
      grad_in.push_back(grad_buff_[output_id.idx]);

      AutodiffMeta<T> grad_out;
      try {
        grad_out = n.apply_backward(grad_in);
      } catch (const std::exception &e) {
        throw std::runtime_error(std::string("apply_backward threw in op ") +
                                 std::string(n.name()) + ": " + e.what());
      }

      FUSION_CHECK(grad_out.size() == inputs.size(),
                   std::string("backward arity mismatch: got ") +
                       std::to_string(grad_out.size()) + " for " +
                       std::to_string(inputs.size()) + " inputs in " +
                       std::string(n.name()));

      for (size_t j = 0; j < inputs.size(); ++j) {
        auto &dst = grad_buff_[inputs[j].idx];
        const auto &src = grad_out.at(j); // bounds-checked
        if (!dst.is_initialised()) {
          dst = src;
        } else {
          FUSION_CHECK(dst.size() == src.size(), "grad size mismatch");
          autodiff::NoGradGuard ng;
          dst = dst + src;
        }
      }
    }
  }

  template <class Op> // TODO: evaluate impl
  ValueID apply(const std::vector<ValueID> &vids) {
    NodeID dst_nid = graph_.template build_node<Op>(AutodiffMeta<T>{});
    AutodiffMeta<T> in;
    in.data.reserve(vids.size());
    auto &node = graph_.nodes[dst_nid.idx];
    for (size_t i = 0; i < vids.size(); i++) {
      NodeID src_nid = graph_.produced_by.at(vids[i].idx).nid;
      graph_.add_edge(src_nid, dst_nid);
      graph_.set_node_input(node, vids[i]);
      graph_.append_consumer_table(dst_nid, vids[i], i);
      // TODO: make output fan out a shared_ptr scheme instead of clones
      in.push_back(val_buff_[vids[i].idx]);
    }
    AutodiffMeta<T> out = run_forward(node, in);
    if (node.outputs.empty()) {
      node.outputs.reserve(out.size());
      for (size_t i = 0; i < out.size(); ++i) {
        ValueID vid = graph_.new_intermediate_value();
        graph_.set_produced_by(vid, dst_nid, static_cast<size_t>(i));
        node.outputs.push_back(vid);
        ensure_value_capacity(vid);
      }
    }
    FUSION_CHECK(out.size() > 0,
                 "Engine::apply: forward produced empty outputs");
    FUSION_BOUNDS_CHECK(0, node.outputs.size());
    FUSION_CHECK(node.outputs.size() == out.size(),
                 "node output size mismatch");
    ValueID out_vid = node.outputs[0];
    ensure_value_capacity(out_vid);
    for (size_t i = 0; i < out.size(); ++i) {
      ValueID vid_i = node.outputs[i];
      ensure_value_capacity(vid_i);
      val_buff_[vid_i.idx] = out[i];
    }
    return node.outputs[0]; // TODO: return all vids here?
  }

  ValueID track_input(Tensor<T>& t) {
    if (t.vid_.idx >= 0) return t.vid_;
    ValueID vid = graph_.new_input_value();
    ensure_value_capacity(vid);
    val_buff_[vid.idx] = t;
    t.vid_ = vid;
    return vid;
  }

  Tensor<T> materialise(ValueID vid) {
    Tensor<T> out = val_buff_[vid.idx];
    out.vid_ = vid;
    return out;
  }



  Tensor<T> get_grad(ValueID vid) {
    FUSION_BOUNDS_CHECK(vid.idx, val_buff_.size());
    return grad_buff_[vid.idx];
   }



  void dump_graph(std::ostream &os) const {
    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
      const auto &n = graph_.nodes[i];
      os << "Node " << i << " [" << n.name() << "]\n";
    }

    const size_t n = std::min(grad_buff_.size(), graph_.produced_by.size());
    for (size_t i = 0; i < n; ++i) {
      const auto &prod = graph_.produced_by[i];
      NodeID nid = prod.nid;
      if (nid.idx >= 0 && static_cast<size_t>(nid.idx) < graph_.nodes.size()) {
        os << "Node idx: " << nid.idx
           << " Node Op: " << graph_.nodes[nid.idx].name() << " ";
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
  std::vector<Tensor<T>> val_buff_;
  std::vector<Tensor<T>> grad_buff_;

  void ensure_value_capacity(ValueID vid) {
    if (val_buff_.size() <= static_cast<size_t>(vid.idx)) {
      val_buff_.resize(static_cast<size_t>(vid.idx) + 1);
    }
  }
  ValueID feed_raw(Tensor<T> &data) {
    ValueID vid = graph_.new_input_value();
    ensure_value_capacity(vid);
    val_buff_[vid.idx] = data;
    return vid;
  }

  template <class Op> ValueID feed(AutodiffMeta<T> v) {
    NodeID dst_nid = graph_.template build_node<Op>(v);
    auto &node = graph_.nodes[dst_nid.idx];
    // Arbitrily set to 0 for single tensor feed
    // TODO: make return type vec<ValueID> to allow for multi output??? Not sure
    // this is needed
    ValueID vid = node.outputs.at(0);
    ensure_value_capacity(vid);
    val_buff_[vid.idx] = v[0];
    return vid;
  }

  void set_grad_buff_size() { grad_buff_.resize(val_buff_.size()); }

  AutodiffMeta<T> grad_init(ValueID vid, size_t out_slot) {
    Tensor<T> grad = ones_like(val_buff_[vid.idx]);
    grad_buff_[vid.idx] = grad;
    AutodiffMeta<T> gradVec;
    gradVec.push_back(grad);
    return gradVec;
  }

  ValueID get_output(std::vector<NodeID> &sorted_nodes, size_t out_slot) {
    std::vector<ValueID> outputs =
        graph_.nodes[sorted_nodes.back().idx].outputs;
    return outputs[out_slot];
  }

  AutodiffMeta<T> run_forward(INode<T> &node, AutodiffMeta<T> &vec) {
    return node.apply_forward(vec);
  }
};

#endif // ENGINE_H
