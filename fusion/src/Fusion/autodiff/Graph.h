#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include "NodeInterface.h"

static constexpr int kNoNode = -1;

struct Edge {
    NodeID src;
    NodeID dst;
  	Edge(NodeID src = NodeID{-1}, NodeID dst = NodeID{-1}) : src(src), dst(dst) {};
};

struct ProducerInfo { NodeID nid; uint16_t out_slot; };

class Graph {
  public:
    Graph() = default;
    std::int16_t node_counter = 0;
    std::uint16_t value_counter = 0;
    std::vector<INode> nodes;
    std::vector<NodeID> node_ids;
    std::vector<Edge> edges;
    std::vector<ProducerInfo> producer_of;

    void add_edge(NodeID src_nid, NodeID dst_nid) {
      if (src_nid.idx == kNoNode || dst_nid.idx == kNoNode) return;
      edges.push_back(Edge{src_nid, dst_nid});
    }


    ValueID new_input_value() {
      ValueID vid{value_counter++};
      if (producer_of.size() <= static_cast<size_t>(vid.idx)) {
        producer_of.resize(static_cast<size_t>(vid.idx) + 1);
    	}
      producer_of[vid.idx] = ProducerInfo{NodeID{kNoNode}, 0};
      return vid;
    };

    template <typename ConcreteOp>
    NodeID build_node() {
      auto op = ConcreteOp{};
      INode node(op);
      uint16_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      auto& stored = nodes.back();
      NodeID nid = make_node_id();
      this->append_producer_table(stored, nid);
      return nid;
  }


  template <class ConcreteOp>
  NodeID build_node(UnaryType<float> vec) {
      auto op = ConcreteOp{};
      INode node(op);
      uint16_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      auto& stored = nodes.back();
      NodeID nid = make_node_id();
      this->append_producer_table(stored, nid);
      return nid;
    }


  template <class ConcreteOp>
  NodeID build_node(BinaryType<float> vec) {
      auto op = ConcreteOp{};
      INode node(op);
      nodes.emplace_back(std::move(node));
      auto& stored = nodes.back();
      NodeID nid = make_node_id();
      this->append_producer_table(stored, nid);
      return nid;
    }



  private:
    NodeID make_node_id() {
      NodeID nid = NodeID{node_counter};
      node_ids.push_back(nid);
      node_counter++;
      return nid;
    }


void append_producer_table(INode& node, NodeID nid) {
    uint16_t num = node.get_static_num_outputs();
    node.outputs.resize(num);
    for (uint16_t i = 0; i < num; i++) {
        ValueID vid{value_counter++};
        node.outputs[i] = vid;

        if (producer_of.size() <= static_cast<size_t>(vid.idx)) {
            producer_of.resize(static_cast<size_t>(vid.idx) + 1);
        }
        producer_of[vid.idx] = ProducerInfo{ nid, i };
    }
}


};


#endif // GRAPH_H
