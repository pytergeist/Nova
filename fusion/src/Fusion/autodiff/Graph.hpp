#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <memory>

#include "ADTypes.h"
#include "AutodiffMeta.hpp"
#include "NodeInterface.hpp"

static constexpr NodeID kNoNode = NodeID{-1};

template <typename T> class Engine;

template <typename T> class Graph {
 public:
   Graph() = default;

   Graph(const Graph &) = delete;
   Graph &operator=(const Graph &) = delete;

   Graph(Graph &&) = delete;
   Graph &operator=(Graph &&) = delete;

   ~Graph() = default;

   std::vector<INode<T>> &nodes() { return nodes_; }

   std::vector<NodeID> &node_ids() { return node_ids_; }
   std::vector<NodeID> node_ids() const { return node_ids_; }

   std::vector<ProducerInfo> &produced_by() { return produced_by_; }
   std::vector<ProducerInfo> produced_by() const { return produced_by_; }

   std::vector<std::vector<ConsumerInfo>> &consumed_by() {
      return consumed_by_;
   }
   std::vector<std::vector<ConsumerInfo>> consumed_by() const {
      return consumed_by_;
   }

   INode<T> &get_node(NodeID id) { return nodes_.at(id); }

   ProducerInfo get_produced_by(ValueID id) const {
      return produced_by_.at(id);
   }

   std::vector<ConsumerInfo> get_consumed_by(ValueID id) const {
      return consumed_by_.at(id);
   }
   std::vector<ConsumerInfo> &get_consumed_by(ValueID id) {
      return consumed_by_.at(id);
   }

 private:
   friend class Engine<T>;

   std::vector<std::vector<ConsumerInfo>> consumed_by_;
   std::vector<INode<T>> nodes_{};
   std::vector<NodeID> node_ids_;
   std::vector<Edge> edges_;
   std::vector<ProducerInfo> produced_by_;
   std::int32_t node_counter_ = 0;
   std::int32_t value_counter_ = 0;

   NodeID make_node_id() {
      NodeID nid = NodeID{node_counter_};
      node_ids_.push_back(nid);
      node_counter_++;
      return nid;
   }

   void set_node_input(INode<T> &node, ValueID vid) { node.add_input(vid); }

   void set_node_output(INode<T> &node, ValueID vid) { node.add_output(vid); }

   ValueID new_input_value() {
      ValueID vid{value_counter_++};
      if (produced_by_.size() <= static_cast<std::size_t>(vid)) {
         produced_by_.resize(static_cast<std::size_t>(vid) + 1);
      }
      produced_by_[vid] = ProducerInfo{.nid = kNoNode, .out_slot = 0};
      return vid;
   };

   ValueID new_intermediate_value() {
      ValueID vid{value_counter_++};

      if (produced_by_.size() <= static_cast<std::size_t>(vid)) {
         produced_by_.resize(static_cast<std::size_t>(vid) + 1);
      }
      if (consumed_by_.size() <= static_cast<std::size_t>(vid)) {
         consumed_by_.resize(static_cast<std::size_t>(vid) + 1);
      }
      return vid;
   }

   void add_edge(NodeID src_nid, NodeID dst_nid) {
      if (src_nid == kNoNode || dst_nid == kNoNode) {
         return;
      }
      edges_.emplace_back(src_nid, dst_nid);
   }

   void append_producer_table(INode<T> &node, NodeID nid) {
      size_t num = node.get_static_num_outputs();
      for (size_t i = 0; i < num; i++) {
         ValueID vid{value_counter_++};
         node.set_output(i, vid);

         if (produced_by_.size() <= static_cast<size_t>(vid)) {
            produced_by_.resize(static_cast<size_t>(vid) + 1);
         }
         produced_by_[vid] = ProducerInfo{.nid = nid, .out_slot = i};
      }
   }

   template <typename ConcreteOp> NodeID build_node() {
      auto op = ConcreteOp{};
      INode<T> node(op);
      nodes_.emplace_back(std::move(node));
      auto &stored = nodes_.back();
      NodeID nid = make_node_id();
      append_producer_table(stored, nid);
      return nid;
   }

   void append_consumer_table(NodeID dst_nid, ValueID vid, size_t slot) {
      if (consumed_by_.size() <= static_cast<size_t>(vid)) {
         consumed_by_.resize(static_cast<size_t>(vid) + 1);
      }
      consumed_by_[vid].push_back(
          ConsumerInfo{.nid = dst_nid, .in_slot = slot});
   }

   void set_produced_by(ValueID vid, NodeID nid, size_t out_slot) {
      if (produced_by_.size() <= static_cast<size_t>(vid)) {
         produced_by_.resize(static_cast<size_t>(vid) + 1);
      }
      produced_by_[vid] = ProducerInfo{.nid = nid, .out_slot = out_slot};
   }
};

#endif // GRAPH_HPP
