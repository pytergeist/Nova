#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include "NodeInterface.h"

struct Edge {
    NodeID v;
    NodeID w;
  	Edge(NodeID v = NodeID{-1}, NodeID w = NodeID{-1}) : v(v), w(w) {};
};


class Graph {
  public:
    Graph() = default;
    std::int16_t node_counter = 0;
    std::uint16_t value_counter = 0;
    std::vector<INode> nodes;
    std::vector<NodeID> node_ids;
    std::vector<Edge> edges;
    std::vector<std::pair<NodeID, ValueID>> producer_of;

    template <class ConcreteOp>
    void build_leaf_node() {
      auto op = ConcreteOp{};
      INode node(op);
      uint16_t num_outputs = node.get_static_num_outputs();
      make_output_ids(node, num_outputs);
      nodes.emplace_back(std::move(node));
      make_node_id();
  }

//  	template <class ConcreteOp>
//    void build_node(INode& node, uint16_t output_idx)) {
//      make_node_id();
//  }

  private:
    void make_node_id() {
      node_ids.push_back(NodeID{node_counter});
      node_counter++ ;
    }

   	void make_output_ids(INode& node, uint16_t num) {
          node.outputs.resize(num);
          for (uint16_t i = 0; i < num; i++) {
            node.outputs[i] = ValueID{value_counter};
            value_counter++;
          }
   	}



};


#endif // GRAPH_H
