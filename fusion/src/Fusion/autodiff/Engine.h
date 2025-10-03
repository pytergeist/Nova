#ifndef ENGINE_H
#define ENGINE_H

#include <any>
#include <memory>

#include "Graph.h"

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
    std::vector<ValueID> vids; // was: std::vector<ValueID> vids(num);
    vids.reserve(num);
    for (size_t i = 0; i < num; i++) {
      vids.push_back(feed_raw(payload[i]));
    }
    return apply<Op>(vids); // what here?
  }

  //    template <class Op>
  //    ValueID apply(UnaryType<T> payload) {
  //        ValueID in_vid = feed_raw(payload.a);
  //        return apply<Op>(in_vid);
  //    }

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
