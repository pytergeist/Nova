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

    std::vector<T> a{1,2,3,4};
    std::vector<T> b{1,2,3,4};

    Engine<T> engine;

    ValueID v1 = engine.apply<AddOp>(BinaryType<T>{a, b});

    ValueID v2 = engine.apply<ExpOp>( v1 );
    ValueID v3 = engine.apply<ExpOp>( v2 );

    const auto& out = engine.value_buffer[v3.idx];
    for (auto x : out) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "Producer info\n";
    for (size_t i = 0; i < engine.graph.producer_of.size(); ++i) {
        auto p = engine.graph.producer_of[i];
        std::cout << p.nid.idx << " " << p.out_slot << "\n";
    }
}
