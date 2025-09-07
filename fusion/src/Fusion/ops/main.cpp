#include <vector>
#include <iostream>
#include "Operation.h"
#include "LinAlg/LinAlg.h"
#include "../autodiff/Node.h"
#include "../autodiff/Graph.h"

#include "../autodiff/NodeInterface.h"



int main() {
    using T = float;
    using Op = Subtract<T>;
    using ConcreteOp = Operation<T, Op>;

    std::vector<T> a{1,2,3,4};
    std::vector<T> b{1,2,3,4};

    auto add_op = ConcreteOp{};
    INode node(add_op);
//    Node<Operation<T, Op>> node(add_op);

    std::any out_any = node.forward(BinaryType<float>{a, b});
    auto y = std::any_cast<ConcreteOp::Out>(std::move(out_any));

    for (auto v: y.a) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

    UnaryType<float> gy;
    gy.a.assign(y.a.size(), 1.0f);

    std::any gy_any = gy;                          // backward takes std::any&
    std::any gx_any = node.backward(gy_any);
    auto gx = std::any_cast<ConcreteOp::GradIn>(std::move(gx_any));

//
    for (auto v : gx.a) std::cout << v << ' ';
    std::cout << std::endl;
    for (auto v : gx.b) std::cout << v << ' ';
    std::cout << std::endl;

}
