#ifndef ENGINE_H
#define ENGINE_H

#include <any>
#include <memory>

#include "Graph.h"
#include "../autodiff/Sort.h"

// TODO: general TODO, move private members of all autodiff classes into private

template <typename T> class Engine {
public:
  std::vector<std::vector<T>> value_buffer;
  std::vector<std::vector<T>> grad_buffer;
  Graph graph{};
  Engine() = default;

  void ensure_value_capacity(ValueID vid) {
    if (value_buffer.size() <= static_cast<size_t>(vid.idx)) {
      value_buffer.resize(static_cast<size_t>(vid.idx) + 1);
    }
  }

  ValueID feed_raw(const std::vector<T> &data) {
    ValueID vid = graph.new_input_value();
    ensure_value_capacity(vid);
    value_buffer[vid.idx] = data;
    return vid;
  }

  template <class Op> ValueID feed(MultiTensor<T> v) {
    NodeID dst_nid = graph.build_node<Op>(v);
    auto &node = graph.nodes[dst_nid.idx];
    // Arbitrily set to 0 for single tensor feed
    // TODO: make return type vec<ValueID> to allow for multi output??? Not sure
    // this is needed
    ValueID vid = node.outputs.at[0];
    ensure_value_capacity(vid);
    value_buffer[vid.idx] = v[0];
    return vid;
  }
  template <class Op> ValueID apply(MultiTensor<T> payload) {
    size_t num = payload.size();
    std::vector<ValueID> vids;
    vids.reserve(num);
    for (size_t i = 0; i < num; i++) {
      vids.push_back(feed_raw(payload[i]));
    }
    return apply<Op>(vids);
  }

  void set_grad_buff_size() { grad_buffer.resize(value_buffer.size()); }

  std::any grad_init(ValueID vid, uint16_t out_slot) {
    std::vector<T> vec = value_buffer[vid.idx];
    std::vector<T> grad(vec.size(), 1);
    std::any gradVec = MultiTensor<T>{grad};
    grad_buffer[vid.idx] = grad;
    return gradVec;
  }

  ValueID get_output(std::vector<NodeID> &sorted_nodes, uint16_t out_slot) {
    std::vector<ValueID> outputs = graph.nodes[sorted_nodes.back().idx].outputs;
    return outputs[out_slot];
  }

  void backward() {
    set_grad_buff_size();
    Sort sort = Sort(graph.nodes.size());
    std::vector<uint16_t> in_degree =
        sort.calc_indegree(graph.nodes, graph.produced_by);
    std::vector<NodeID> sorted_nodes = sort.topological_sort(
        graph.nodes, graph.produced_by, graph.consumed_by, graph.node_ids);
    ValueID out_vid = get_output(sorted_nodes, 0);
    std::any gradVec = grad_init(out_vid, 0);
    for (int16_t i = sorted_nodes.size() - 1; i > -1; --i) {
      auto &n = graph.nodes[sorted_nodes[i].idx];
      auto &inputs = n.inputs;
      auto output_id = n.outputs[0];
      if (grad_buffer[output_id.idx].empty())
        continue;
      gradVec = MultiTensor<T>{grad_buffer[output_id.idx]};
      gradVec = n.apply_backward(gradVec);
      auto grad = std::any_cast<MultiTensor<T>>(gradVec);
      for (uint16_t j = 0; j < inputs.size(); ++j) {
        grad_buffer[inputs[j].idx] = grad[j];
      }
    }
  }

  template <class Op> // TODO: evaluate impl
  ValueID apply(std::vector<ValueID> vids) {
    NodeID dst_nid = graph.build_node<Op>(MultiTensor<T>{});
    MultiTensor<T> in; // was: MultiTensor<T> in(vids.size());
    in.data.reserve(vids.size());
    auto &node = graph.nodes[dst_nid.idx];
    for (uint16_t i = 0; i < vids.size(); i++) {
      NodeID src_nid = graph.produced_by.at(vids[i].idx).nid;
      graph.add_edge(src_nid, dst_nid);
      graph.set_node_input(node, vids[i]);
      graph.append_consumer_table(dst_nid, vids[i], i);
      in.push_back(value_buffer[vids[i].idx]);
    }
    std::any any_out = run_forward(node, std::any{in});
    auto out_mt = std::any_cast<typename Op::Out>(any_out);
    if (node.outputs.empty()) {
      node.outputs.reserve(out_mt.size());
      for (size_t i = 0; i < out_mt.size(); ++i) {
        ValueID vid = graph.new_intermediate_value();
        graph.set_produced_by(vid, dst_nid, static_cast<uint16_t>(i));
        node.outputs.push_back(vid);
        ensure_value_capacity(vid);
      }
    }

    if (out_mt.size() == 0) {
      throw std::runtime_error("No outputs produced");
    }
    ValueID out_vid = node.outputs[0];
    ensure_value_capacity(out_vid);
    auto &out_u = std::any_cast<MultiTensor<T> &>(any_out);
    value_buffer[out_vid.idx] = out_u[0];
    return out_vid;
  }

  std::any run_forward(INode &node, const std::any &vec) {
    return node.apply_forward(vec);
  }
};

#endif // ENGINE_H
