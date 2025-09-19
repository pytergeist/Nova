#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include "NodeInterface.h"

struct Edge {
    NodeID v;
    NodeID w;
  	Edge(NodeID v = NodeID{-1}, NodeID w = NodeID{-1}) : v(v), w(w) {};
};

struct ProducerInfo { NodeID node; ValueID out_slot; };

class Graph {
  public:
    Graph() = default;
    std::int16_t node_counter = 0;
    std::uint16_t value_counter = 0;
    std::vector<INode> nodes;
    std::vector<NodeID> node_ids;
    std::vector<Edge> edges;
    std::vector<ProducerInfo> producer_of;

    template <typename ConcreteOp>
    void build_node() {
      auto op = ConcreteOp{};
      INode node(op);
      uint16_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      NodeID nid = make_node_id();
  }


  template <class ConcreteOp>
  NodeID build_node(UnaryType<float> vec) {
      auto op = ConcreteOp{};
      INode node(op);
      uint16_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      auto& stored = nodes.back();
      NodeID nid = make_node_id();
      append_producer_table(stored, nid);
      return nid;
    }


  template <class ConcreteOp>
  NodeID build_node(BinaryType<float> vec) {
      auto op = ConcreteOp{};
      INode node(op);
      nodes.emplace_back(std::move(node));
      auto& stored = nodes.back();
      NodeID nid = make_node_id();
      append_producer_table(stored, nid);
      return nid;
    }



  private:
    NodeID make_node_id() {
      NodeID nid = NodeID{node_counter};
      node_ids.push_back(nid);
      node_counter++;
      return nid;
    }

  void append_producer_table(INode& node, NodeID nid) {
      uint16_t num = node.get_static_num_outputs();
      node.outputs.resize(num);
      for (uint16_t i = 0; i < num; i++) {
        ValueID vid = ValueID{value_counter};
        node.outputs[i] = vid;
        producer_of.push_back(ProducerInfo{nid, vid});
        value_counter++;
      }
    }

};


#endif // GRAPH_H
