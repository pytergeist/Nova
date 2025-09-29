#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include "NodeInterface.h"

static constexpr int16_t kNoNode = -1;

struct Edge {
    NodeID src;
    NodeID dst;
  	Edge(NodeID src = NodeID{-1}, NodeID dst = NodeID{-1}) : src(src), dst(dst) {};
};

struct ProducerInfo { NodeID nid; uint16_t out_slot; };

struct ConsumerInfo { NodeID nid; uint16_t in_slot; };

class Graph {
  public:
    Graph() = default;
    std::int16_t node_counter = 0;
    std::uint16_t value_counter = 0;
    std::vector<INode> nodes;
    std::vector<NodeID> node_ids;
    std::vector<Edge> edges;
    std::vector<ProducerInfo> produced_by;
    std::vector<std::vector<ConsumerInfo>> consumed_by;

    void add_edge(NodeID src_nid, NodeID dst_nid) {
      if (src_nid.idx == kNoNode || dst_nid.idx == kNoNode) return;
      edges.push_back(Edge{src_nid, dst_nid});
    }


    ValueID new_input_value() {
      ValueID vid{value_counter++};
      if (produced_by.size() <= static_cast<size_t>(vid.idx)) {
        produced_by.resize(static_cast<size_t>(vid.idx) + 1);
    	}
      produced_by[vid.idx] = ProducerInfo{NodeID{kNoNode}, 0};
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

  void set_node_inputs(INode& node, std::vector<ValueID> vids) {
  node.inputs.resize(vids.size());
  for (uint16_t i = 0; i < vids.size(); i++) {
    node.inputs[i] = vids[i];
  }
}
  void append_consumer_table(NodeID dst_nid, std::vector<ValueID> vids) {
    	if (consumed_by.size() <= static_cast<size_t>(value_counter)) {
          consumed_by.resize(consumed_by.size() + vids.size());
    	}
        consumed_by.resize(consumed_by.size() + vids.size());
        for (uint16_t i = 0; i < vids.size(); i++) {
          consumed_by[vids[i].idx].push_back(ConsumerInfo{dst_nid, i});;
      }
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

        if (produced_by.size() <= static_cast<size_t>(vid.idx)) {
            produced_by.resize(static_cast<size_t>(vid.idx) + 1);
        }
        produced_by[vid.idx] = ProducerInfo{ nid, i };
    }
}


};


#endif // GRAPH_H
