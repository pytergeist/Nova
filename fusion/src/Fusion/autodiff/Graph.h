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
      producer_of.push_back(ProducerInfo{NodeID{node_counter}, ValueID{value_counter}});
      make_node_id();
      make_output_ids(node, num_outputs);
  }


  template <class ConcreteOp>
  UnaryType<float> build_node(UnaryType<float> vec) {
      auto op = ConcreteOp{};
      INode node(op);
      uint16_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      auto& stored = nodes.back();
      producer_of.push_back(ProducerInfo{NodeID{node_counter}, ValueID{value_counter}});
      make_node_id();
      make_output_ids(stored, num_outputs);
      UnaryType<float> y = this->tmp_run_forward<ConcreteOp>(stored, vec);
      return y;
    }


  template <class ConcreteOp>
  UnaryType<float> build_node(BinaryType<float> vec) {
      auto op = ConcreteOp{};
      INode node(op);
      uint16_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      auto& stored = nodes.back();
      producer_of.push_back(ProducerInfo{NodeID{node_counter}, ValueID{value_counter}});
      make_node_id();
      make_output_ids(stored, num_outputs);
      UnaryType<float> y = this->tmp_run_forward<ConcreteOp>(stored, vec);
      return y;
    }


  // this will abstracted into the engine
  template <class ConcreteOp>
  UnaryType<float> tmp_run_forward(INode &node, BinaryType<float> vec) {
      UnaryType<float> y = node.forward_t<ConcreteOp>(vec);
      return y;
  }
  // this will abstracted into the engine
  template <class ConcreteOp>
  UnaryType<float> tmp_run_forward(INode &node, UnaryType<float> vec) {
      UnaryType<float> y = node.forward_t<ConcreteOp>(vec);
      return y;
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
