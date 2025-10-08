#include <iostream>
#include <vector>
#include <cstddef>
#include <vector>

#include "autodiff/Engine.h"
#include "autodiff/Graph.h"
#include "autodiff/Node.h"
#include "autodiff/NodeInterface.h"
#include "autodiff/Sort.h"
#include "policies/LinAlg/LinAlg.h"
#include "policies/Transcendental/Transcendental.h"
#include "policies/Operation.h"
#include "Tensor.h"

int main() {
    using T = float;
    using AddOp = Operation<T, Add<T>>;
//    using ExpOp = Operation<T, Exp<T>>;
//    using divOp = Operation<T, Divide<T>>;
//    using MulOp = Operation<T, Multiply<T>>;
//    using subOp = Operation<T, Subtract<T>>;
//    using logOp = Operation<T, Log<T>>;
//    using sqrtOp = Operation<T, Sqrt<T>>;
//    using powOp = Operation<T, Pow<T>>;

    std::vector<T> v1{1, 2, 3, 4};
    std::vector<T> v2{1, 2, 3, 4};

    std::size_t shape = 4;

	Tensor<T> a{{shape}, v1};
    Tensor<T> b{{shape}, v2};

    Engine<T> engine;

    MultiTensor<T> mt1;
    mt1.push_back(std::move(a));
    mt1.push_back(std::move(b));

    ValueID v0 = engine.apply<AddOp>(std::move(mt1));
//    ValueID v1 = engine.apply<ExpOp>(std::vector<ValueID>{v0});
//    ValueID v2 = engine.apply<MulOp>(MultiTensor<T>{std::move(a), std::move(b)});
//    ValueID v3 = engine.apply<AddOp>(std::vector<ValueID>{v1, v2});
//    ValueID v4 = engine.apply<AddOp>(MultiTensor<T>{std::move(a), std::move(b)});
//    ValueID v5 = engine.apply<ExpOp>(std::vector<ValueID>{v4});
//    ValueID v6 = engine.apply<MulOp>(std::vector<ValueID>{v3, v5});
//    ValueID v7 = engine.apply<divOp>(std::vector<ValueID>{v5, v6});
//    ValueID v8 = engine.apply<subOp>(std::vector<ValueID>{v6, v7});
//    ValueID v9 = engine.apply<logOp>(std::vector<ValueID>{v8});
//    ValueID v10 = engine.apply<sqrtOp>(std::vector<ValueID>{v9});
//    ValueID v11 = engine.apply<sqrtOp>(std::vector<ValueID>{v10});
//
//
//    std::cout << "v1: " << v1.idx << std::endl;
//    std::cout << "v2: " << v2.idx << std::endl;
//    std::cout << "v3: " << v3.idx << std::endl;
//    std::cout << "v4: " << v4.idx << std::endl;
//    std::cout << "v5: " << v5.idx << std::endl;
//    std::cout << "v6: " << v6.idx << std::endl;
//    std::cout << "v7: " << v7.idx << std::endl;
//    std::cout << "v8: " << v8.idx << std::endl;

//    engine.backward();

    engine.dump_graph(std::cout);
}
