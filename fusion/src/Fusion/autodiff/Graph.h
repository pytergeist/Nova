#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include "NodeInterface.h"

struct ValueId {uint16_t idx;};
struct NodeId {uint16_t idx;};


class Graph {
  public:
    Graph() = default;
    std::uint16_t node_counter = 0;
    std::uint16_t value_counter = 0;
    std::vector<INode> nodes;
    std::vector<NodeId> node_ids;
    std::vector<ValueId> value_ids;

    void add_node(INode&& node, uint16_t num_outputs) {
//      node.outputs.resize(num_outputs);
      nodes.emplace_back(std::move(node));
      create_node_idx();
  }

  private:
    void create_node_idx() {
      node_ids.emplace_back(node_counter);
      node_counter++;
    }

    void create_value_idx() {
		value_ids.emplace_back(value_counter);
		value_counter++;
    }


};


#endif // GRAPH_H
