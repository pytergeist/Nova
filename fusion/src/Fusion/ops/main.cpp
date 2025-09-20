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
    using Op1 = Add<T>;
    using ConcreteOp1 = Operation<T, Op1>;
    using Op2 = Exp<T>;
    using ConcreteOp2 = Operation<T, Op2>;
	using Op3 = Exp<T>;
	using ConcreteOp3 = Operation<T, Op3>;

    std::vector<T> a{1,2,3,4};
    std::vector<T> b{1,2,3,4};

	BinaryType<float> v{a, b};

    Engine<T> engine{};

    std::any x = engine.apply<ConcreteOp1>(v);
    auto& u = std::any_cast<UnaryType<T>&>(x);
    std::any y = engine.apply<ConcreteOp2>(u);
    auto& w = std::any_cast<UnaryType<T>&>(y);
    std::any z = engine.apply<ConcreteOp3>(w);

    for (uint16_t i = 0; i < engine.value_buffer.size(); i++) {
      for (auto v : engine.value_buffer[i]) {
        std::cout << v << " ";
      }
      std::cout << std::endl;
    }




//    engine.graph.build_node<ConcreteOp1>();
//    engine.graph.build_node<ConcreteOp2>();
//	engine.graph.build_node<ConcreteOp3>();
//
//	for (uint16_t i = 0; i < engine.graph.nodes.size(); i++) {
//		auto& n = engine.graph.nodes[i];
//		v = engine.run_forward(n, v);
//		auto& u = std::any_cast<UnaryType<T>&>(v);
//        std::cout << typeid(u).name() << std::endl;
//
//        for (auto n : u.a) {
//			std::cout << n << " ";
//        }
//	}



}
