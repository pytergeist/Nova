#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include "NodeInterface.h"


class Graph {
  public:
    Graph() = default;
    std::vector<INode> created_nodes;

    void add_node(INode&& node) {
      created_nodes.emplace_back(std::move(node));
  }
};


#endif // GRAPH_H
