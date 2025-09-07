#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include "Node.h"


template <class NodeType>
class Graph {
  public:
    Graph() = default;
    std::vector<NodeType> created_nodes;

    void add_node(NodeType node) {
      created_nodes.push_back(node);
    }
};


#endif // GRAPH_H
