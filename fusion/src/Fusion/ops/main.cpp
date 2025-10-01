#include <vector>
#include <iostream>
#include "Operation.h"
#include "LinAlg/LinAlg.h"
#include "Transcendental/Exp.h"
#include "../autodiff/Node.h"
#include "../autodiff/Graph.h"
#include "../autodiff/NodeInterface.h"
#include "../autodiff/Engine.h"
#include "../autodiff/Sort.h"

int main() {
    using T = float;
    using AddOp = Operation<T, Add<T>>;
    using ExpOp = Operation<T, Exp<T>>;
    using divOp = Operation<T, Divide<T>>;
    using MulOp = Operation<T, Multiply<T>>;

    std::vector<T> a{1,2,3,4};
    std::vector<T> b{1,2,3,4};

    Engine<T> engine;

	ValueID v0 = engine.apply<AddOp>(BinaryType<T>{a, b});
	ValueID v1 = engine.apply<ExpOp>(v0);
	ValueID v2 = engine.apply<MulOp>(BinaryType<T>{a, b});
	ValueID v3 = engine.apply<AddOp>(v1, v2);
	ValueID v4 = engine.apply<ExpOp>(engine.apply<AddOp>(BinaryType<T>{a, b}));
	ValueID v5 = engine.apply<MulOp>(v3, v4);

    std::cout << "v1: " << v1.idx << std::endl;
    std::cout << "v2: " << v2.idx << std::endl;
    std::cout << "v3: " << v3.idx << std::endl;
    std::cout << "v4: " << v4.idx << std::endl;
    std::cout << "v5: " << v5.idx << std::endl;


    const auto& out = engine.value_buffer[v4.idx];
    for (auto x : out) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "Produced By info\n";
    for (size_t i = 0; i < engine.graph.produced_by.size(); ++i) {
        auto p = engine.graph.produced_by[i];
        std::cout << "ValueID: " << i << " " << "NodeID: " << p.nid.idx << " " << "out_slot: " << p.out_slot;
        if (p.nid.idx != -1) {
          std::cout << " Test ValueID: " << engine.graph.nodes[p.nid.idx].outputs[p.out_slot].idx << "\n";
          }
        else {
        std::cout << "\n";
        }
    }

  std::cout << "Consumed By info\n";
  for (size_t i = 0; i < engine.graph.consumed_by.size(); ++i) {
    auto p = engine.graph.consumed_by[i];
    for (auto x : p) {
      std::cout << "ValueID: " << i << " " << "NodeID: " << x.nid.idx << " " << "in_slot: " << x.in_slot;
      std::cout << " Test ValueID: " << engine.graph.nodes[x.nid.idx].inputs[x.in_slot].idx << "\n";
    }
  }

    std::cout << "Edge List\n";
    for (size_t i = 0; i < engine.graph.edges.size(); ++i) {
      auto p = engine.graph.edges[i];
      std::cout << p.src.idx << " " << p.dst.idx << "\n";
    }

    std::cout << "Node info\n";
    for (size_t i = 0; i < engine.graph.nodes.size(); ++i) {
      auto& p = engine.graph.nodes[i];
      std::cout << "Node idx: " << i << " Input idxs: ";
      for (auto x : p.inputs)
        std::cout << x.idx << ", ";
      std::cout << std::endl;

      std::cout << "Node idx: " << i << " Output idxs: ";
      for (auto x : p.outputs)
        std::cout << x.idx << ", ";
     std::cout << std::endl;
	}

   std::cout << "Sorting\n";

   Sort sort = Sort(engine.graph.nodes.size());

   std::vector<uint16_t> in_degree = sort.calc_indegree(engine.graph.nodes, engine.graph.produced_by);
   std::cout << "In degree\n";
   for (uint16_t i = 0; i < in_degree.size(); ++i) {
     std::cout << "Node idx: " << i << " Node Degree:  " << in_degree[i] << "\n";
   }

   std::cout << "Sorted idx's\n";
   std::vector<NodeID> sorted_nodes = sort.topological_sort(
       engine.graph.nodes,
       engine.graph.produced_by,
       engine.graph.consumed_by,
       engine.graph.node_ids
       );

   for (auto x : sorted_nodes) {
     std::cout << "Node idx: " << x.idx << "\n";
   }

  std::cout << "Reverse Order\n";
  for (uint16_t i = sorted_nodes.size() - 1; i > 0; --i) {
    std::cout << "Node idx: " << i << "\n";
    if (i == 6) {
      auto& out = engine.graph.nodes[sorted_nodes[i].idx].outputs;
      for (auto x : out) {
        std::cout << "Output ValueID: " << x.idx << "\n";
      }
	auto val = engine.value_buffer[sorted_nodes[i].idx];
    for (auto x : val) {
      std::cout << x << " ";
    }
    std::cout << "\n";
    }
  }

  std::cout << std::endl;
  std::cout << std::endl;

  std::cout << "Initialising Grad Vector\n";
  auto val = engine.value_buffer[sorted_nodes[6].idx];
  std::vector<T> grad(val.size(), 1);
  for (auto x : grad) {
    std::cout << x << " ";
  }
  std::cout << std::endl;
//  UnaryType<T> unaryGrad{grad};
  std::any grad_vec = UnaryType<T>{grad};
//
//  // -------------
//  auto& n5 = engine.graph.nodes[sorted_nodes[5].idx];
//  grad_vec = n5.apply_backward(grad_vec);
//
//  auto& n4 = engine.graph.nodes[sorted_nodes[4].idx];
//  std::cout << n4.name() << std::endl;
//  grad_vec = n4.apply_backward(grad_vec); // Breaking type mismatch
  uint16_t slot_idx = 0;
  std::vector<ValueID> inputs;
  std::vector<ValueID> outputs;
  outputs = engine.graph.nodes[sorted_nodes[sorted_nodes.size() - 1].idx].outputs;
  auto output = outputs[slot_idx];
  auto initial = engine.value_buffer[output.idx];
  std::vector<T> initialGrad(initial.size(), 1);
  std::any gradVec = UnaryType<T>{initialGrad};
  engine.grad_buffer.resize(engine.value_buffer.size());
  std::cout << "Vec size: " << engine.value_buffer.size() << "\n";
  std::cout << "Vec idx: " << output.idx << "\n";
  engine.grad_buffer[output.idx] = UnaryType<T>{initialGrad};
  for (int16_t i = sorted_nodes.size() - 1; i > -1; --i) {
      auto& n = engine.graph.nodes[sorted_nodes[i].idx];
      auto& inputs = n.inputs;
      auto output_id = n.outputs[0];
      gradVec = engine.grad_buffer[output_id.idx];
      // here we have final node (id = 6), output valueID = 12
      // we have inputs list, two slots [ValueID, ValueID] = [7, 11]
      // we can get dst nodes with the produced by table - this gives
      // nid1 = 3, nid2 = 5

      gradVec = n.apply_backward(gradVec);
      if (n.grad_in_type() == typeid(BinaryType<T>)) {
        auto grad = std::any_cast<BinaryType<T>>(gradVec);
		engine.grad_buffer[inputs[0].idx] = UnaryType<T>{grad.a};
       	engine.grad_buffer[inputs[1].idx] = UnaryType<T>{grad.b};
      }
      else if (n.grad_out_type() == typeid(UnaryType<T>)) {
        auto grad = std::any_cast<UnaryType<T>>(gradVec);
        engine.grad_buffer[inputs[0].idx] = UnaryType<T>{grad.a};
      }
  }

  for (uint16_t i = 0; i < engine.grad_buffer.size(); ++i) {
    std::cout << "Node idx: " << i << " ";
    for (auto x : engine.grad_buffer[i].a) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  }

//    auto& n = engine.graph.nodes[sorted_nodes[i].idx];
//    std::cout << "Node: " << sorted_nodes[i].idx << " Op: " << n.name();
//    std::cout << " GradInType: " << n.grad_in_type().name();
//    std::cout << " GradOutType: " << n.grad_out_type().name() << std::endl;
//    grad_vec = n.apply_backward(grad_vec);
//    std::cout << " GradVec: " << typeid(grad_vec).name() << std::endl;;
  }
