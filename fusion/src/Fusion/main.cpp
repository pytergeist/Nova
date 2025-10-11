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
#include "storage/TensorBuffer.h"

int main() {
    using T = float;
    using AddOp = Operation<T, Add<T>>;
    using ExpOp = Operation<T, Exp<T>>;
    using divOp = Operation<T, Divide<T>>;
    using MulOp = Operation<T, Multiply<T>>;
    using subOp = Operation<T, Subtract<T>>;
    using logOp = Operation<T, Log<T>>;
    using sqrtOp = Operation<T, Sqrt<T>>;
    using powOp = Operation<T, Pow<T>>;

    std::vector<T> va{1, 2, 3, 4};
    std::vector<T> vb{1, 2, 3, 4};
    std::vector<T> vc{1, 2, 3, 4};
    std::vector<T> vd{1, 2, 3, 4};
    std::vector<T> ve{1, 2, 3, 4};
    std::vector<T> vf{1, 2, 3, 4};

    std::size_t shape = 4;

	Tensor<T> a{{shape}, va};
    Tensor<T> b{{shape}, vb};
    Tensor<T> c{{shape}, vc};
    Tensor<T> d{{shape}, vd};
    Tensor<T> e{{shape}, vc};
    Tensor<T> f{{shape}, vd};


    Engine<T> engine;

    MultiTensor<T> mt1;
    mt1.push_back(std::move(a));
    mt1.push_back(std::move(b));

    MultiTensor<T> mt2;
    mt2.push_back(std::move(c));
    mt2.push_back(std::move(d));

//    for (auto &t : mt2) {
//      std::cout << t << std::endl;
//    }
//
//
    MultiTensor<T> mt3;
    mt3.push_back(std::move(e));
    mt3.push_back(std::move(f));
//
//
// ----- Broken Ops: Mul, Log
    ValueID v0 = engine.apply<AddOp>(std::move(mt1));
    ValueID v1 = engine.apply<ExpOp>(std::vector<ValueID>{v0});
    ValueID v2 = engine.apply<AddOp>(std::move(mt2));
    ValueID v3 = engine.apply<MulOp>(std::vector<ValueID>{v0, v1}); // was MulOp
    ValueID v4 = engine.apply<AddOp>(std::move(mt3));
    ValueID v5 = engine.apply<ExpOp>(std::vector<ValueID>{v4});
    ValueID v6 = engine.apply<MulOp>(std::vector<ValueID>{v3, v5}); // was MulOp
    ValueID v7 = engine.apply<divOp>(std::vector<ValueID>{v5, v6});
    ValueID v8 = engine.apply<subOp>(std::vector<ValueID>{v6, v7});
    ValueID v9 = engine.apply<ExpOp>(std::vector<ValueID>{v8}); // Was log
//    ValueID v10 = engine.apply<sqrtOp>(std::vector<ValueID>{v9});
//    ValueID v11 = engine.apply<powOp>(std::vector<ValueID>{v10, v9});

//
//    std::cout << "v1: " << v1.idx << std::endl;
//    std::cout << "v2: " << v2.idx << std::endl;
//    std::cout << "v3: " << v3.idx << std::endl;
//    std::cout << "v4: " << v4.idx << std::endl;
//    std::cout << "v5: " << v5.idx << std::endl;
//    std::cout << "v6: " << v6.idx << std::endl;
//    std::cout << "v7: " << v7.idx << std::endl;
//    std::cout << "v8: " << v8.idx << std::endl;
//
    engine.backward();
//
    engine.dump_graph(std::cout);
//
//
//    std::vector<T> vtb{1, 2, 3, 4};
//
//    TensorBuffer buff = TensorBuffer::allocate_elements<T>(vtb.size());
//    buff.copy_from<T>(vtb, 0);
//    std::cout << "buff type: " << typeid(buff).name() << std::endl;
//    std::cout << "buff size_bytes: " << buff.size_bytes() << std::endl;
//    std::cout << "buff size (elems): " << buff.size<T>() << std::endl;
//    std::cout << "Use count: " << buff.use_count() << std::endl;
//
//    TensorBuffer buff2 = buff;
//    std::cout << "buff Use count: " << buff.use_count() << std::endl;Tensor<T> b{{shape}, vb};









}
