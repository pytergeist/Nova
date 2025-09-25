#include <vector>
#include <iostream>
#include "Operation.h"
#include "LinAlg/LinAlg.h"
#include "Transcendental/Exp.h"
#include "../autodiff/Node.h"
#include "../autodiff/Graph.h"
#include "../autodiff/NodeInterface.h"
#include "../autodiff/Engine.h"

int main() {
    using T = float;
    using AddOp = Operation<T, Add<T>>;
    using ExpOp = Operation<T, Exp<T>>;
    using divOp = Operation<T, Divide<T>>;
    using MulOp = Operation<T, Multiply<T>>;

    std::vector<T> a{1,2,3,4};
    std::vector<T> b{1,2,3,4};

    Engine<T> engine;

    ValueID v1 = engine.apply<AddOp>(BinaryType<T>{a, b});

    ValueID v2 = engine.apply<ExpOp>( v1 );
    ValueID v3 = engine.apply<Divide<T>>(v1, v2);
    ValueID v4 = engine.apply<MulOp>(v2, v3);

    std::cout << "v1: " << v1.idx << std::endl;
    std::cout << "v2: " << v2.idx << std::endl;
    std::cout << "v3: " << v3.idx << std::endl;
    std::cout << "v4: " << v4.idx << std::endl;


    const auto& out = engine.value_buffer[v4.idx];
    for (auto x : out) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "Producer info\n";
    for (size_t i = 0; i < engine.graph.producer_of.size(); ++i) {
        auto p = engine.graph.producer_of[i];
        std::cout << p.nid.idx << " " << p.out_slot << "\n";
    }

    std::cout << "Edge List\n";
    for (size_t i = 0; i < engine.graph.edges.size(); ++i) {
      auto p = engine.graph.edges[i];
      std::cout << p.src.idx << " " << p.dst.idx << "\n";
    }

    std::cout << "Node info\n";
    for (size_t i = 0; i < engine.graph.nodes.size(); ++i) {
      auto& p = engine.graph.nodes[i];
      std::cout << "Node idx: " << i << " Input idxs: ";
      for (auto x : p.inputs)
        std::cout << x.idx << ", ";
      std::cout << std::endl;

      std::cout << "Node idx: " << i << " Output idxs: ";
      for (auto x : p.outputs)
        std::cout << x.idx << ", ";
     std::cout << std::endl;
	}
}
