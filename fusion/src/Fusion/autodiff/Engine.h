#ifndef ENGINE_H
#define ENGINE_H

#include <any>
#include <memory>

#include "Graph.h"
#include "Sort.h"
#include "Traits.h"

// TODO: general TODO, move private members of all autodiff classes into private

template <typename T> class Engine {
public:
  Engine() : graph_{}, val_buff_{}, grad_buff_{} {};

  template <class Op> ValueID apply(MultiTensor<T> payload) {
    size_t num = payload.size();
    std::vector<ValueID> vids;
    vids.reserve(num);
    for (size_t i = 0; i < num; i++) {
      vids.push_back(feed_raw(payload[i]));
    }
    return apply<Op>(vids);
  }

  void backward() {
    set_grad_buff_size();
    Sort sort_(graph_.nodes.size());
    std::vector<uint16_t> in_degree =
        sort_.calc_indegree(graph_.nodes, graph_.produced_by);
    std::vector<NodeID> sorted_nodes = sort_.topological_sort(
        graph_.nodes, graph_.produced_by, graph_.consumed_by, graph_.node_ids);
    ValueID out_vid = get_output(sorted_nodes, 0);
    std::any gradVec = grad_init(out_vid, 0);
    for (int16_t i = sorted_nodes.size() - 1; i > -1; --i) {
      auto &n = graph_.nodes[sorted_nodes[i].idx];
      auto &inputs = n.inputs;
      auto output_id = n.outputs[0];
      if (grad_buff_[output_id.idx].empty())
        continue;
      gradVec = MultiTensor<T>{grad_buff_[output_id.idx]};
      gradVec = n.apply_backward(gradVec);
      auto grad = std::any_cast<MultiTensor<T>>(gradVec);
      for (uint16_t j = 0; j < inputs.size(); ++j) {
        grad_buff_[inputs[j].idx] = grad[j];
      }
    }
  }

  template <class Op> // TODO: evaluate impl
  ValueID apply(std::vector<ValueID> vids) {
    NodeID dst_nid = graph_.build_node<Op>(MultiTensor<T>{});
    MultiTensor<T> in;
    in.data.reserve(vids.size());
    auto &node = graph_.nodes[dst_nid.idx];
    for (uint16_t i = 0; i < vids.size(); i++) {
      NodeID src_nid = graph_.produced_by.at(vids[i].idx).nid;
      graph_.add_edge(src_nid, dst_nid);
      graph_.set_node_input(node, vids[i]);
      graph_.append_consumer_table(dst_nid, vids[i], i);
      in.push_back(val_buff_[vids[i].idx]);
    }
    std::any any_out = run_forward(node, std::any{in});
    auto out_mt = std::any_cast<typename Op::Out>(any_out);
    if (node.outputs.empty()) {
      node.outputs.reserve(out_mt.size());
      for (size_t i = 0; i < out_mt.size(); ++i) {
        ValueID vid = graph_.new_intermediate_value();
        graph_.set_produced_by(vid, dst_nid, static_cast<uint16_t>(i));
        node.outputs.push_back(vid);
        ensure_value_capacity(vid);
      }
    }
    FUSION_CHECK(out_mt.size() != 0,
                 "Engine::apply: forward produced empty outputs");
    ValueID out_vid = node.outputs[0];
    ensure_value_capacity(out_vid);
    auto &out_u = std::any_cast<MultiTensor<T> &>(any_out);
    val_buff_[out_vid.idx] = out_u[0];
    return out_vid;
  }

  void dump_graph(std::ostream &os) const {
    for (size_t i = 0; i < graph_.nodes.size(); ++i) {
      const auto &n = graph_.nodes[i];
      os << "Node " << i << " [" << n.name() << "]\n";
    }
    for (uint16_t i = 0; i < grad_buff_.size(); ++i) {
      NodeID nid = graph_.produced_by[i].nid;
      if (nid.idx >= 0) {
        os << "Node idx: " << nid.idx
           << " Node Op: " << graph_.nodes[nid.idx].name() << " ";
        if (grad_buff_.at(i).empty()) {
          os << "[no grad]" << std::endl;
        }
        for (auto x : grad_buff_.at(i)) {
          os << x << " ";
        }
        os << std::endl;
      }
    }
  }

private:
  Graph graph_{};
  std::vector<std::vector<T>> val_buff_;
  std::vector<std::vector<T>> grad_buff_;

  void ensure_value_capacity(ValueID vid) {
    if (val_buff_.size() <= static_cast<size_t>(vid.idx)) {
      val_buff_.resize(static_cast<size_t>(vid.idx) + 1);
    }
  }
  ValueID feed_raw(const std::vector<T> &data) {
    ValueID vid = graph_.new_input_value();
    ensure_value_capacity(vid);
    val_buff_[vid.idx] = data;
    return vid;
  }

  template <class Op> ValueID feed(MultiTensor<T> v) {
    NodeID dst_nid = graph_.build_node<Op>(v);
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

  std::any grad_init(ValueID vid, uint16_t out_slot) {
    std::vector<T> vec = val_buff_[vid.idx];
    std::vector<T> grad(vec.size(), 1);
    std::any gradVec = MultiTensor<T>{grad};
    grad_buff_[vid.idx] = grad;
    return gradVec;
  }

  ValueID get_output(std::vector<NodeID> &sorted_nodes, uint16_t out_slot) {
    std::vector<ValueID> outputs =
        graph_.nodes[sorted_nodes.back().idx].outputs;
    return outputs[out_slot];
  }

  std::any run_forward(INode &node, const std::any &vec) {
    return node.apply_forward(vec);
  }
};

#endif // ENGINE_H
