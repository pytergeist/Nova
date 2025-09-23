#ifndef ENGINE_H
#define ENGINE_H


#include <memory>
#include "Graph.h"

template <typename T>
class Engine {
  public:
    // You need to store inputs and outputs
    std::vector<std::vector<T>> value_buffer;
    Graph graph{};
    Engine() = default;

  // Change name of this method
  void add_value(UnaryType<T> v) {
      value_buffer.push_back(v.a);
  };

  // Change name of this method
  void add_value(BinaryType<T> v) {
	value_buffer.push_back(v.a);
    value_buffer.push_back(v.b);
  };


  template <class Op>
  std::any apply(BinaryType<T> a) {
    this->add_value(a);
    this->graph.build_node<Op>();
    auto& n = this->graph.nodes.back();
    std::any v = a;
    v = this->run_forward(n, v);
    return v;
  };

  template <class Op>
  std::any apply(UnaryType<T> a) {
    this->add_value(a);
    this->graph.build_node<Op>();
    auto& n = this->graph.nodes.back();
    std::any v = a;
    v = this->run_forward(n, v);
    return v;
  };


  std::any run_forward(INode &node, const std::any& vec) {
      return node.apply_forward(vec);
  }

};



#endif // ENGINE_H
