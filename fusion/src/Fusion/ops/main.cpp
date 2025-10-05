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
    using subOp = Operation<T, Subtract<T>>;

    std::vector<T> a{1,2,3,4};
    std::vector<T> b{1,2,3,4};

    Engine<T> engine;

	ValueID v0 = engine.apply<AddOp>(MultiTensor<T>{a, b});
	ValueID v1 = engine.apply<ExpOp>(std::vector<ValueID>{v0});
	ValueID v2 = engine.apply<MulOp>(MultiTensor<T>{a, b});
	ValueID v3 = engine.apply<AddOp>(std::vector<ValueID>{v1, v2});
    ValueID v4 = engine.apply<AddOp>(MultiTensor<T>{a, b});
	ValueID v5 = engine.apply<ExpOp>(std::vector<ValueID>{v4});
	ValueID v6 = engine.apply<MulOp>(std::vector<ValueID>{v3, v4});
    ValueID v7 = engine.apply<divOp>(std::vector<ValueID>{v3, v4});
    ValueID v8 = engine.apply<subOp>(std::vector<ValueID>{v3, v4});

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
  uint16_t slot_idx = 0;
  std::vector<ValueID> inputs;
  std::vector<ValueID> outputs;
  outputs = engine.graph.nodes[sorted_nodes[sorted_nodes.size() - 1].idx].outputs;
  auto output = outputs[slot_idx];
  auto initial = engine.value_buffer[output.idx];
  std::vector<T> initialGrad(initial.size(), 1);
  std::any gradVec = MultiTensor<T>{initialGrad};
  engine.grad_buffer.resize(engine.value_buffer.size());
  std::cout << "Vec size: " << engine.value_buffer.size() << "\n";
  std::cout << "Vec idx: " << output.idx << "\n";
  engine.grad_buffer[output.idx] = initialGrad;
  for (int16_t i = sorted_nodes.size() - 1; i > -1; --i) {
      auto& n = engine.graph.nodes[sorted_nodes[i].idx];
      auto& inputs = n.inputs;
      auto output_id = n.outputs[0];
      gradVec = MultiTensor{engine.grad_buffer[output_id.idx]};
      gradVec = n.apply_backward(gradVec);
      for (uint16_t j = 0; j < inputs.size(); ++j) {
        auto grad = std::any_cast<MultiTensor<T>>(gradVec);
        engine.grad_buffer[inputs[j].idx] = grad[j];
      }
  }

  for (uint16_t i = 0; i < engine.grad_buffer.size(); ++i) {
    NodeID nid = engine.graph.produced_by[i].nid;
    if (nid.idx >= 0) {
    std::cout << "Node idx: " << nid.idx << " Node Op: " << engine.graph.nodes[nid.idx].name() << " ";
    for (auto x : engine.grad_buffer[i]) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  }
  }

  }
