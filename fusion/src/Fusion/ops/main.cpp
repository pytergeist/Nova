#include "../autodiff/Engine.h"
#include "../autodiff/Graph.h"
#include "../autodiff/Node.h"
#include "../autodiff/NodeInterface.h"
#include "../autodiff/Sort.h"
#include "LinAlg/LinAlg.h"
#include "Operation.h"
#include "Transcendental/Exp.h"
#include <iostream>
#include <vector>

int main() {
  using T = float;
  using AddOp = Operation<T, Add<T>>;
  using ExpOp = Operation<T, Exp<T>>;
  using divOp = Operation<T, Divide<T>>;
  using MulOp = Operation<T, Multiply<T>>;
  using subOp = Operation<T, Subtract<T>>;

  std::vector<T> a{1, 2, 3, 4};
  std::vector<T> b{1, 2, 3, 4};

  Engine<T> engine;

  ValueID v0 = engine.apply<AddOp>(MultiTensor<T>{a, b});
  ValueID v1 = engine.apply<ExpOp>(std::vector<ValueID>{v0});
  ValueID v2 = engine.apply<MulOp>(MultiTensor<T>{a, b});
  ValueID v3 = engine.apply<AddOp>(std::vector<ValueID>{v1, v2});
  ValueID v4 = engine.apply<AddOp>(MultiTensor<T>{a, b});
  ValueID v5 = engine.apply<ExpOp>(std::vector<ValueID>{v4});
  ValueID v6 = engine.apply<MulOp>(std::vector<ValueID>{v3, v5});
  ValueID v7 = engine.apply<divOp>(std::vector<ValueID>{v5, v6});
  ValueID v8 = engine.apply<subOp>(std::vector<ValueID>{v6, v7});

  std::cout << "v1: " << v1.idx << std::endl;
  std::cout << "v2: " << v2.idx << std::endl;
  std::cout << "v3: " << v3.idx << std::endl;
  std::cout << "v4: " << v4.idx << std::endl;
  std::cout << "v5: " << v5.idx << std::endl;
  std::cout << "v6: " << v6.idx << std::endl;
  std::cout << "v7: " << v7.idx << std::endl;
  std::cout << "v8: " << v8.idx << std::endl;

  engine.backward();

  engine.dump_graph(std::cout);
}
