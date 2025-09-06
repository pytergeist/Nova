#include <vector>
#include <iostream>
#include "Operation.h"
#include "LinAlg/Registry.cpp"
#include "../Node.h"



int main() {
    std::vector<float> a{1,2,3,4};
    std::vector<float> b{1,2,3,4};

    auto context = std::make_shared<Context<float>>();
    auto add_op  = std::make_shared<Operation<float, Divide<float>>>();

    Node<float, Operation<float, Divide<float>>, Context<float>> node(add_op, context);

    auto y = node.op->forward(*context, BinaryType<float>{a, b});

    for (auto v: y.a) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

    UnaryType<float> gy;
    gy.a.assign(y.a.size(), 1.0f);

    auto gx = node.op->backward(*context, gy);

    for (auto v : gx.a) std::cout << v << ' ';
    std::cout << std::endl;
    for (auto v : gx.b) std::cout << v << ' ';
    std::cout << std::endl;






}
