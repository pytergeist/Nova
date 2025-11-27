#ifndef GRAPH_H
#define GRAPH_H

#include <memory>

#include "NodeInterface.h"
#include "Traits.h"

static constexpr int16_t kNoNode = -1;

struct Edge {
   NodeID src;
   NodeID dst;
   Edge(NodeID src = NodeID{-1}, NodeID dst = NodeID{-1})
       : src(src), dst(dst) {};
};

struct ProducerInfo {
   NodeID nid;
   size_t out_slot;
};

struct ConsumerInfo {
   NodeID nid;
   size_t in_slot;
};

template <typename T> class Graph {
 public:
   Graph() = default;
   std::int32_t node_counter = 0;
   std::int32_t value_counter = 0;
   std::vector<INode<T>> nodes;
   std::vector<NodeID> node_ids;
   std::vector<Edge> edges;
   std::vector<ProducerInfo> produced_by;
   std::vector<std::vector<ConsumerInfo>> consumed_by;

   Graph(const Graph &) = delete;
   Graph &operator=(const Graph &) = delete;
   Graph(Graph &&) = delete;
   Graph &operator=(Graph &&) = delete;

   void add_edge(NodeID src_nid, NodeID dst_nid) {
      if (src_nid.idx == kNoNode || dst_nid.idx == kNoNode)
         return;
      edges.push_back(Edge{src_nid, dst_nid});
   }

   ValueID new_intermediate_value() {
      ValueID vid{value_counter++};

      if (produced_by.size() <= static_cast<size_t>(vid.idx)) {
         produced_by.resize(static_cast<size_t>(vid.idx) + 1);
      }
      if (consumed_by.size() <= static_cast<size_t>(vid.idx)) {
         consumed_by.resize(static_cast<size_t>(vid.idx) + 1);
      }
      return vid;
   }

   void set_produced_by(ValueID vid, NodeID nid, size_t out_slot) {
      if (produced_by.size() <= static_cast<size_t>(vid.idx)) {
         produced_by.resize(static_cast<size_t>(vid.idx) + 1);
      }
      produced_by[vid.idx] = ProducerInfo{nid, out_slot};
   }

   ValueID new_input_value() {
      ValueID vid{value_counter++};
      if (produced_by.size() <= static_cast<size_t>(vid.idx)) {
         produced_by.resize(static_cast<size_t>(vid.idx) + 1);
      }
      produced_by[vid.idx] = ProducerInfo{NodeID{kNoNode}, 0};
      return vid;
   };

   template <typename ConcreteOp> NodeID build_node() {
      auto op = ConcreteOp{};
      INode<T> node(op);
      size_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      auto &stored = nodes.back();
      NodeID nid = make_node_id();
      this->append_producer_table(stored, nid);
      return nid;
   }

   template <typename ConcreteOp> NodeID build_node(AutodiffMeta<T> &vec) {
      auto op = ConcreteOp{};
      INode<T> node(op);
      size_t num_outputs = node.get_static_num_outputs();
      nodes.emplace_back(std::move(node));
      auto &stored = nodes.back();
      NodeID nid = make_node_id();
      this->append_producer_table(stored, nid);
      return nid;
   }

   void set_node_input(INode<T> &node, ValueID vid) {
      size_t curr_size = node.inputs.size();
      node.inputs.resize(curr_size + 1);
      node.inputs[curr_size] = vid;
   }

   void append_consumer_table(NodeID dst_nid, ValueID vid, size_t slot) {
      if (consumed_by.size() <= static_cast<size_t>(vid.idx)) {
         consumed_by.resize(static_cast<size_t>(vid.idx) + 1);
      }
      consumed_by[vid.idx].push_back(ConsumerInfo{dst_nid, slot});
   }

 private:
   NodeID make_node_id() {
      NodeID nid = NodeID{node_counter};
      node_ids.push_back(nid);
      node_counter++;
      return nid;
   }

   void append_producer_table(INode<T> &node, NodeID nid) {
      size_t num = node.get_static_num_outputs();
      node.outputs.resize(num);
      for (size_t i = 0; i < num; i++) {
         ValueID vid{value_counter++};
         node.outputs[i] = vid;

         if (produced_by.size() <= static_cast<size_t>(vid.idx)) {
            produced_by.resize(static_cast<size_t>(vid.idx) + 1);
         }
         produced_by[vid.idx] = ProducerInfo{nid, i};
      }
   }
};

#endif // GRAPH_H
