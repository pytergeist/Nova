#include <vector>
#include <iostream>
#include "Operation.h"
#include "LinAlg/LinAlg.h"
#include "../autodiff/Node.h"



int main() {
    std::vector<float> a{1,2,3,4};
    std::vector<float> b{1,2,3,4};

    auto add_op =  Operation<float, Subtract<float>>();

    Node<float, Operation<float, Subtract<float>>> node(add_op);

    auto y = node.run_forward(BinaryType<float>{a, b});

    for (auto v: y.a) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

    UnaryType<float> gy;
    gy.a.assign(y.a.size(), 1.0f);
//
    auto gx = node.run_backward(gy);
//
    for (auto v : gx.a) std::cout << v << ' ';
    std::cout << std::endl;
    for (auto v : gx.b) std::cout << v << ' ';
    std::cout << std::endl;






}
