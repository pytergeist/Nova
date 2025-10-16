#include <iostream>
#include <vector>
#include <cstddef>
#include <vector>

#include "autodiff/Engine.h"
#include "autodiff/Graph.h"
#include "autodiff/Node.h"
#include "autodiff/NodeInterface.h"
#include "autodiff/Sort.h"
#include "autodiff/policies/LinAlg/LinAlg.h"
#include "autodiff/policies/Transcendental/Transcendental.h"
#include "autodiff/policies/Reduction/Reduction.h"
#include "autodiff/policies/Reduction/Sum.h"
#include "autodiff/policies/Operation.h"
#include "Tensor.h"
#include "storage/TensorBuffer.h"
#include "autodiff/policies/Comparison/Comparison.h"
#include "autodiff/policies/Shape/Shape.h"
#include "autodiff/policies/Ewise/Ewise.h"
#include "autodiff/EngineContext.h"



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
    using sumOp = Operation<T, Sum<T>>;
    using maximumOp = Operation<T, Maximum<T>>;
    using greaterthanOp = Operation<T, GreaterThan<T>>;
    using transposeOp = Operation<T, Transpose<T>>;
    using matmulOp = Operation<T, MatMul<T>>;


    std::vector<T> va{1, 2, 3, 4};
    std::vector<T> vb{1, 2, 3, 4};
    std::vector<T> vc{1, 2, 3, 4};
    std::vector<T> vd{1, 2, 3, 4};
    std::vector<T> ve{1, 2, 3, 4};
    std::vector<T> vf{1, 2, 3, 4};

    std::vector<std::size_t> shape = {2,2};


	Tensor<T> a{shape, va, Device::CPU, true};
    Tensor<T> b{shape, vb, Device::CPU, true};
    Tensor<T> c{shape, vc, Device::CPU, true};
    Tensor<T> d{shape, vd, Device::CPU, true};
    Tensor<T> e{shape, vc, Device::CPU, true};
    Tensor<T> f{shape, vd, Device::CPU, true};


    Engine<T> engine;
    EngineContext<T>::set(&engine);
//
    Tensor<T> g = a + b;
    Tensor<T> h = c * d;
    Tensor<T> i = g / h;
    Tensor<T> j = h.matmul(i);
//    Tensor<T> j = g - i;
//    Tensor<T> k = i / j;
//
//    std::cout << g << std::endl;
//
//    std::cout << a.requires_grad_ << std::endl;
//    std::cout << b.requires_grad_ << std::endl;
//    std::cout << c.requires_grad_ << std::endl;
//    std::cout << d.requires_grad_ << std::endl;
    std::cout << g << std::endl;
    std::cout << h << std::endl;
    std::cout << i << std::endl;

//     std::cout << i << std::endl;

//
     engine.backward();

     engine.dump_graph(std::cout);




//    MultiTensor<T> mt1;
//    mt1.push_back(a);
//    mt1.push_back(b);
//
//    MultiTensor<T> mt2;
//    mt2.push_back(c);
//    mt2.push_back(d);
//
//
//
//
//
//    MultiTensor<T> mt3;
//    mt3.push_back(e);
//    mt3.push_back(f);
//
//
//    ValueID v0 = engine.apply<AddOp>(std::move(mt1));
////    ValueID v1 = engine.apply<sqrtOp>(std::vector<ValueID>{v0});
//    ValueID v2 = engine.apply<AddOp>(std::move(mt2));
//    ValueID v3 = engine.apply<divOp>(std::vector<ValueID>{v0, v2}); // was MulOp
////    ValueID v4 = engine.apply<AddOp>(std::move(mt3));
////    ValueID v5 = engine.apply<sqrtOp>(std::vector<ValueID>{v4});
////    ValueID v6 = engine.apply<MulOp>(std::vector<ValueID>{v3, v5}); // was MulOp
////    ValueID v7 = engine.apply<divOp>(std::vector<ValueID>{v5, v6});
////    ValueID v8 = engine.apply<subOp>(std::vector<ValueID>{v6, v7});
////    ValueID v9 = engine.apply<sqrtOp>(std::vector<ValueID>{v8}); // Was log
////    ValueID v10 = engine.apply<sqrtOp>(std::vector<ValueID>{v9});
////    ValueID v11 = engine.apply<powOp>(std::vector<ValueID>{v10, v9});
////    ValueID v12 = engine.apply<logOp>(std::vector<ValueID>{v11});
////    ValueID v13 = engine.apply<matmulOp>(std::vector<ValueID>{v11, v12});
////    ValueID v14 = engine.apply<maximumOp>(std::vector<ValueID>{v12, v13});
////    ValueID v15 = engine.apply<greaterthanOp>(std::vector<ValueID>{v13, v14});
////    ValueID v16 = engine.apply<transposeOp>(std::vector<ValueID>{v15});
//
//
//
//    engine.backward();
//
//    engine.dump_graph(std::cout);
////
////
////    std::vector<T> vtb{1, 2, 3, 4};
////
////    TensorBuffer buff = TensorBuffer::allocate_elements<T>(vtb.size());
////    buff.copy_from<T>(vtb, 0);
////    std::cout << "buff type: " << typeid(buff).name() << std::endl;
////    std::cout << "buff size_bytes: " << buff.size_bytes() << std::endl;
////    std::cout << "buff size (elems): " << buff.size<T>() << std::endl;
////    std::cout << "Use count: " << buff.use_count() << std::endl;
////
////    TensorBuffer buff2 = buff;
////    std::cout << "buff Use count: " << buff.use_count() << std::endl;Tensor<T> b{{shape}, vb};
//
//
//






}
