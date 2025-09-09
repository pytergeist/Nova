#include <vector>
#include <iostream>
#include "Operation.h"
#include "LinAlg/LinAlg.h"
#include "Transcendental/Exp.h"
#include "../autodiff/Node.h"
#include "../autodiff/Graph.h"

#include "../autodiff/NodeInterface.h"


//    Node<Operation<T, Op>> node(add_op);

//    std::any out_any = node.forward(BinaryType<float>{a, b});
//    auto y = std::any_cast<ConcreteOp::Out>(std::move(out_any));



//    std::any gy_any = gy;                          // backward takes std::any
//    std::any gx_any = node.backward(gy_any);
//    auto gx = std::any_cast<ConcreteOp::GradIn>(std::move(gx_any));

//

int main() {
    using T = float;
    using Op1 = Exp<T>;
    using ConcreteOp1 = Operation<T, Op1>;
    using Op2 = Add<T>;
    using ConcreteOp2 = Operation<T, Op2>;

    std::vector<T> a{1,2,3,4};
    std::vector<T> b{1,2,3,4};

    auto exp_op = ConcreteOp1{};
    auto add_op = ConcreteOp2{};

    Graph graph{};
    INode node1(exp_op);

    INode node2(add_op);

    std::cout << node1.get_static_num_outputs() << std::endl;
    std::cout << node1.get_static_num_inputs() << std::endl;


    std::cout << node2.get_static_num_outputs() << std::endl;
    std::cout << node2.get_static_num_inputs() << std::endl;



    graph.add_node(std::move(node1), 1);


    UnaryType<float> y = graph.nodes[0].forward_t<ConcreteOp1>(UnaryType<float>{a});


    for (auto v: y.a) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

    UnaryType<float> gy;
    gy.a.assign(y.a.size(), 1.0f);
    UnaryType<float> gx = graph.nodes[0].backward_t<ConcreteOp1>(gy);;


    for (auto v : gx.a) std::cout << v << ' ';
    std::cout << std::endl;
//    for (auto v : gx.b) std::cout << v << ' ';
//    std::cout << std::endl;

}
