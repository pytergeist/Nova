#include <vector>
#include <iostream>
#include "Operation.h"
#include "LinAlg/LinAlg.h"
#include "Transcendental/Exp.h"
#include "../autodiff/Node.h"
#include "../autodiff/Graph.h"

#include "../autodiff/NodeInterface.h"


int main() {
    using T = float;
    using Op1 = Add<T>;
    using ConcreteOp1 = Operation<T, Op1>;
    using Op2 = Exp<T>;
    using ConcreteOp2 = Operation<T, Op2>;

    std::vector<T> a{1,2,3,4};
    std::vector<T> b{1,2,3,4};

    auto add_op = ConcreteOp1{};
    auto exp_op = ConcreteOp2{};

    Graph graph{};
    INode node1(add_op);

    INode node2(exp_op);

	std::cout << "Node1<Add> inputs: " << node1.get_static_num_inputs();
    std::cout << " Outputs: " << node1.get_static_num_outputs() << std::endl;

    std::cout << "Node2<Exp> inputs: " << node2.get_static_num_outputs();
    std::cout << " Outputs: " << node2.get_static_num_outputs() << std::endl;


    graph.add_node(std::move(node1), node1.get_static_num_outputs(), node1.get_static_num_inputs());
    graph.add_node(std::move(node2), node2.get_static_num_outputs(), node2.get_static_num_inputs());

    UnaryType<float> y1 = graph.nodes[0].forward_t<ConcreteOp1>(BinaryType<float>{a, b});
    UnaryType<float> y2 = graph.nodes[1].forward_t<ConcreteOp2>(y1);

    std::cout << "Graph<Node> indexs: ";
    std::cout << graph.node_ids[0].idx << ", ";
    std::cout << graph.node_ids[1].idx << std::endl;




    std::cout << "Node1<Add> input ValueIDs: ";
    for (auto v : graph.nodes[0].inputs) {
      std::cout << v.idx << " ";
    }
    std::cout << std::endl;

        std::cout << "Node2<Exp> input ValueIDs: ";
    for (auto v : graph.nodes[1].inputs) {
      std::cout << v.idx << " ";
    }
    std::cout << std::endl;


        std::cout << "Node1<Add> output ValueIDs: ";
    for (auto v : graph.nodes[0].outputs) {
      std::cout << v.idx << " ";
    }
    std::cout << std::endl;

        std::cout << "Node2<Exp> output ValueIDs: ";
    for (auto v : graph.nodes[1].outputs) {
      std::cout << v.idx << " ";
    }
    std::cout << std::endl;

	std::cout << "Forward<Add> Output: ";
    for (auto v: y1.a) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

    std::cout << "Forward<Exp> Output: ";
    for (auto v: y2.a) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

//
//    UnaryType<float> gy;
//    gy.a.assign(y1.a.size(), 1.0f);
//    UnaryType<float> gx = graph.nodes[0].backward_t<ConcreteOp1>(gy);;
//
//
//    for (auto v : gx.a) std::cout << v << ' ';
//    std::cout << std::endl;
//    for (auto v : gx.b) std::cout << v << ' ';
//    std::cout << std::endl;

}
