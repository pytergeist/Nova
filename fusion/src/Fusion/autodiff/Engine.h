#ifndef ENGINE_H
#define ENGINE_H

#include <memory>
#include <any>

#include "Graph.h"

template <typename T>
class Engine {
public:
  std::vector<std::vector<T>> value_buffer;
  Graph graph{};
  Engine() = default;

  void ensure_value_capacity(ValueID vid) {
    if (value_buffer.size() <= static_cast<size_t>(vid.idx)) {
      value_buffer.resize(static_cast<size_t>(vid.idx) + 1);
    }
  }

  ValueID feed_raw(const std::vector<T>& data) {
    ValueID vid = graph.new_input_value();
    ensure_value_capacity(vid);
    value_buffer[vid.idx] = data;
    return vid;
  }

  template <class Op> ValueID feed(UnaryType<T> v) {
    NodeID dst_nid = graph.build_node<Op>(v);
    auto &node = graph.nodes[dst_nid.idx];
    ValueID vid = node.outputs.at[0];
    ensure_value_capacity(vid);
    value_buffer[vid.idx] = v.a;
    return vid;
  }
    template <class Op>
    ValueID apply(BinaryType<T> payload) {
        ValueID a_vid = feed_raw(payload.a);
        ValueID b_vid = feed_raw(payload.b);
        return apply<Op>(a_vid, b_vid);
    }

    template <class Op>
    ValueID apply(UnaryType<T> payload) {
        ValueID in_vid = feed_raw(payload.a);
        return apply<Op>(in_vid);
    }

    template <class Op>
    ValueID apply(ValueID in_vid) {
        NodeID dst_nid = graph.build_node<Op>(UnaryType<T>{});
        NodeID src_nid = graph.produced_by.at(in_vid.idx).nid;

        graph.add_edge(src_nid, dst_nid);
        auto& node = graph.nodes[dst_nid.idx];


        graph.set_node_inputs(node, std::vector<ValueID>{in_vid});
        graph.append_consumer_table(dst_nid, std::vector<ValueID>{in_vid});
        NodeID src = graph.produced_by.at(in_vid.idx).nid;

        UnaryType<T> in{ value_buffer[in_vid.idx] };

        std::any any_out = run_forward(node, std::any{in});

        if (node.outputs.empty()) {
            throw std::runtime_error("No outputs produced");
        }
        ValueID out_vid = node.outputs[0];
        ensure_value_capacity(out_vid);
        auto& out_u = std::any_cast<UnaryType<T>&>(any_out);
        value_buffer[out_vid.idx] = out_u.a;
        return out_vid;
    }

    template <class Op>
    ValueID apply(ValueID a_vid, ValueID b_vid) {
        NodeID dst_nid = graph.build_node<Op>(BinaryType<T>{});
        NodeID src_nida = graph.produced_by.at(a_vid.idx).nid;
        NodeID src_nidb = graph.produced_by.at(b_vid.idx).nid;
        graph.add_edge(src_nida, dst_nid);
        graph.add_edge(src_nidb, dst_nid);
        auto& node = graph.nodes[dst_nid.idx];


        graph.set_node_inputs(node, std::vector<ValueID>{a_vid, b_vid});
        graph.append_consumer_table(dst_nid, std::vector<ValueID>{a_vid, b_vid});


        BinaryType<T> in{ value_buffer[a_vid.idx], value_buffer[b_vid.idx] };

        std::any any_out = run_forward(node, std::any{in});

        if (node.outputs.empty()) {
        	throw std::runtime_error("No outputs produced");
        }
        ValueID out_vid = node.outputs[0];
        ensure_value_capacity(out_vid);
        auto& out_u = std::any_cast<UnaryType<T>&>(any_out);
        value_buffer[out_vid.idx] = out_u.a;
        return out_vid;
    }

  std::any run_forward(INode &node, const std::any &vec) {
    return node.apply_forward(vec);
  }
};

#endif // ENGINE_H
