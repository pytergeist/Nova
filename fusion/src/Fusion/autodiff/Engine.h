#ifndef ENGINE_H
#define ENGINE_H


#include <memory>
#include "Graph.h"

template <typename T>
class Engine {
  public:
    std::vector<std::vector<T>> value_buffer;
    Graph graph{};
    Engine() = default;


  std::any run_forward(INode &node, const std::any& vec) {
      return node.apply_forward(vec);
  }

};



#endif // ENGINE_H
