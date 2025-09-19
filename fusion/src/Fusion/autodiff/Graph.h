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
    void build_leaf_node() {
      auto op = ConcreteOp{};
      INode node(op);
      uint16_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      producer_of.push_back(ProducerInfo{NodeID{node_counter}, ValueID{value_counter}});
      make_node_id();
      make_output_ids(node, num_outputs);
  }


  template <class ConcreteOp1, class ConcreteOp2>
  std::tuple<UnaryType<float>, UnaryType<float>>
  build_node(INode &node1, INode &node2, std::vector<float> a,
             std::vector<float> b) {
        UnaryType<float> y1 = node1.forward_t<ConcreteOp1>(BinaryType<float>{a, b});
        UnaryType<float> y2 = node2.forward_t<ConcreteOp2>(y1);
        return std::make_tuple(y1, y2);
      }

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
