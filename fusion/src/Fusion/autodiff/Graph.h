#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include "NodeInterface.h"


class Graph {
  public:
    Graph() = default;
    std::uint16_t node_counter = 0;
    std::uint16_t value_counter = 0;
    std::vector<INode> nodes;
    std::vector<NodeID> node_ids;

    void add_node(INode&& node, uint16_t num_outputs, uint16_t num_inputs) {
      make_output_ids(node, num_outputs);
      make_input_ids(node, num_inputs);
      nodes.emplace_back(std::move(node));
      make_node_id();
  }

  private:
    void make_node_id() {
      node_ids.emplace_back(node_counter);
      node_counter++;
    }

   	void make_output_ids(INode& node, uint16_t num) {
          node.outputs.resize(num);
          for (uint16_t i = 0; i < num; i++) {
            node.outputs[i] = ValueID{i};
          }
   	}

    void make_input_ids(INode& node, uint16_t num) {
      	  node.inputs.resize(num);
          for (uint16_t i = 0; i < num; i++) {
            node.outputs[i] = ValueID{i};
          }
   	}



};


#endif // GRAPH_H
