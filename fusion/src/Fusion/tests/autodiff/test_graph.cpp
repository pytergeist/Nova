#include <gtest/gtest.h>

#include "helpers.h"

#include "Fusion/autodiff/Graph.hpp"

template <typename T>
class GraphHarness {
public:
    Graph<T> graph;

    NodeID make_node_id() {
        return graph.make_node_id();
    }

    ValueID new_input_value() {
        return graph.new_input_value();
    }

    ValueID new_intermediate_value() {
        return graph.new_intermediate_value();
    }

    void add_edge(NodeID src_nid, NodeID dst_nid) {
        graph.add_edge(src_nid, dst_nid);
    }

    template <typename ConcreteOp>
    NodeID build_node() {
        return graph.template build_node<ConcreteOp>();
    }

    void append_consumer_table(NodeID dst_nid, ValueID vid, size_t slot) {
        graph.append_consumer_table(dst_nid, vid, slot);
    }

    void set_produced_by(ValueID vid, NodeID nid, size_t out_slot) {
        graph.set_produced_by(vid, nid, out_slot);
    }

    void set_node_input(INode<T>& node, ValueID vid) {
        graph.set_node_input(node, vid);
    }

    void set_node_output(INode<T>& node, ValueID vid) {
        graph.set_node_output(node, vid);
    }

    const auto& edges() const {
        return graph.edges_;
    }
};

class GraphTest : public ::testing::Test {
protected:
    using T = float;
    GraphHarness<T> h;
};


TEST_F(GraphTest, default_constructed_graph_is_empty) {
    EXPECT_TRUE(h.graph.nodes().empty());
    EXPECT_TRUE(h.graph.node_ids().empty());
    EXPECT_TRUE(h.graph.produced_by().empty());
    EXPECT_TRUE(h.graph.consumed_by().empty());
    EXPECT_TRUE(h.edges().empty());
}


TEST_F(GraphTest, make_node_id_returns_sequential_ids) {
    NodeID id0 = h.make_node_id();
    NodeID id1 = h.make_node_id();
    NodeID id2 = h.make_node_id();

    EXPECT_EQ(id0, NodeID{0});
    EXPECT_EQ(id1, NodeID{1});
    EXPECT_EQ(id2, NodeID{2});
}

TEST_F(GraphTest, make_node_id_appends_to_node_ids_table) {
    NodeID id0 = h.make_node_id();
    NodeID id1 = h.make_node_id();

    const std::vector<NodeID> ids = h.graph.node_ids();
    ASSERT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], id0);
    EXPECT_EQ(ids[1], id1);
}

TEST_F(GraphTest, new_input_value_creates_producer_entry_with_no_node) {
    ValueID vid = h.new_input_value();

    ProducerInfo producer = h.graph.get_produced_by(vid);
    EXPECT_EQ(producer.nid, kNoNode);
    EXPECT_EQ(producer.out_slot, 0);
}

TEST_F(GraphTest, new_input_value_resizes_produced_by_table) {
    ValueID vid0 = h.new_input_value();
    ValueID vid1 = h.new_input_value();

    const std::vector<ProducerInfo> produced_by = h.graph.produced_by();
    ASSERT_GE(produced_by.size(), 2);

    EXPECT_LT(static_cast<size_t>(vid0), produced_by.size());
    EXPECT_LT(static_cast<size_t>(vid1), produced_by.size());
}

TEST_F(GraphTest, new_intermediate_value_resizes_produced_by_and_consumed_by_tables) {
    ValueID vid = h.new_intermediate_value();

    EXPECT_LT(static_cast<size_t>(vid), h.graph.produced_by().size());
    EXPECT_LT(static_cast<size_t>(vid), h.graph.consumed_by().size());
}

TEST_F(GraphTest, new_values_are_sequential_across_input_and_intermediate_allocations) {
    ValueID v0 = h.new_input_value();
    ValueID v1 = h.new_intermediate_value();
    ValueID v2 = h.new_input_value();

    EXPECT_EQ(v0, ValueID{0});
    EXPECT_EQ(v1, ValueID{1});
    EXPECT_EQ(v2, ValueID{2});
}

TEST_F(GraphTest, append_consumer_table_adds_single_consumer) {
    ValueID vid = h.new_intermediate_value();
    NodeID dst = NodeID{7};

    h.append_consumer_table(dst, vid, 2);

    const std::vector<ConsumerInfo> consumers = h.graph.get_consumed_by(vid);
    ASSERT_EQ(consumers.size(), 1);
    EXPECT_EQ(consumers[0].nid, dst);
    EXPECT_EQ(consumers[0].in_slot, 2);
}

TEST_F(GraphTest, append_consumer_table_appends_multiple_consumers_for_same_value) {
    ValueID vid = h.new_intermediate_value();

    h.append_consumer_table(NodeID{3}, vid, 0);
    h.append_consumer_table(NodeID{5}, vid, 1);

    const std::vector<ConsumerInfo> consumers = h.graph.get_consumed_by(vid);
    ASSERT_EQ(consumers.size(), 2);

    EXPECT_EQ(consumers[0].nid, NodeID{3});
    EXPECT_EQ(consumers[0].in_slot, 0);

    EXPECT_EQ(consumers[1].nid, NodeID{5});
    EXPECT_EQ(consumers[1].in_slot, 1);
}

TEST_F(GraphTest, append_consumer_table_resizes_consumed_by_if_needed) {
    ValueID vid = ValueID{4};

    h.append_consumer_table(NodeID{1}, vid, 0);

    ASSERT_GT(h.graph.consumed_by().size(), static_cast<size_t>(vid));
    ASSERT_EQ(h.graph.get_consumed_by(vid).size(), 1);
}

TEST_F(GraphTest, set_produced_by_stores_producer_info) {
    ValueID vid = ValueID{4};

    h.set_produced_by(vid, NodeID{2}, 1);

    ProducerInfo producer = h.graph.get_produced_by(vid);
    EXPECT_EQ(producer.nid, NodeID{2});
    EXPECT_EQ(producer.out_slot, 1);
}

TEST_F(GraphTest, set_produced_by_resizes_produced_by_if_needed) {
    ValueID vid = ValueID{6};

    h.set_produced_by(vid, NodeID{9}, 0);

    ASSERT_GT(h.graph.produced_by().size(), static_cast<size_t>(vid));

    ProducerInfo producer = h.graph.get_produced_by(vid);
    EXPECT_EQ(producer.nid, NodeID{9});
    EXPECT_EQ(producer.out_slot, 0);
}

TEST_F(GraphTest, add_edge_ignores_no_node_source) {
    h.add_edge(kNoNode, NodeID{1});
    EXPECT_TRUE(h.edges().empty());
}

TEST_F(GraphTest, add_edge_ignores_no_node_destination) {
    h.add_edge(NodeID{1}, kNoNode);
    EXPECT_TRUE(h.edges().empty());
}

TEST_F(GraphTest, add_edge_stores_valid_edge) {
    h.add_edge(NodeID{1}, NodeID{2});
    ASSERT_EQ(h.edges().size(), 1);
    EXPECT_EQ(h.edges()[0].src, NodeID{1});
    EXPECT_EQ(h.edges()[0].dst, NodeID{2});
}

TEST_F(GraphTest, build_node_adds_node_to_storage) {
    NodeID nid = h.build_node<Operation<float, TestUnaryOp<float>>>();

    ASSERT_EQ(h.graph.nodes().size(), 1);
    ASSERT_EQ(h.graph.node_ids().size(), 1);
    EXPECT_EQ(h.graph.node_ids()[0], nid);
}

TEST_F(GraphTest, build_node_allocates_sequential_node_ids) {
    NodeID n0 = h.build_node<Operation<float, TestUnaryOp<float>>>();
    NodeID n1 = h.build_node<Operation<float, TestBinaryOp<float>>>();

    EXPECT_EQ(n0, NodeID{0});
    EXPECT_EQ(n1, NodeID{1});
}

TEST_F(GraphTest, build_unary_node_registers_one_produced_output) {
    NodeID nid = h.build_node<Operation<float, TestUnaryOp<float>>>();

    INode<float>& node = h.graph.get_node(nid);

    ASSERT_EQ(node.outputs().size(), 1);

    ValueID out0 = node.outputs()[0];
    ProducerInfo producer = h.graph.get_produced_by(out0);

    EXPECT_EQ(producer.nid, nid);
    EXPECT_EQ(producer.out_slot, 0);
}

TEST_F(GraphTest, build_split_node_registers_all_produced_outputs) {
    NodeID nid = h.build_node<Operation<float, TestSplitOp<float>>>();

    INode<float>& node = h.graph.get_node(nid);

    ASSERT_EQ(node.outputs().size(), 2);

    ValueID out0 = node.outputs()[0];
    ValueID out1 = node.outputs()[1];

    ProducerInfo p0 = h.graph.get_produced_by(out0);
    ProducerInfo p1 = h.graph.get_produced_by(out1);

    EXPECT_EQ(p0.nid, nid);
    EXPECT_EQ(p0.out_slot, 0);

    EXPECT_EQ(p1.nid, nid);
    EXPECT_EQ(p1.out_slot, 1);
}

TEST_F(GraphTest, build_node_grows_produced_by_table) {
    h.build_node<Operation<float, TestUnaryOp<float>>>();
    const std::size_t size_after_first = h.graph.produced_by().size();

    h.build_node<Operation<float, TestSplitOp<float>>>();
    const std::size_t size_after_second = h.graph.produced_by().size();

    EXPECT_GT(size_after_first, 0);
    EXPECT_GT(size_after_second, size_after_first);
}


TEST_F(GraphTest, set_node_input_adds_input_to_node) {
    NodeID nid = h.build_node<Operation<float, TestUnaryOp<float>>>();
    INode<float>& node = h.graph.get_node(nid);

    ValueID vid = h.new_input_value();
    h.set_node_input(node, vid);

    ASSERT_EQ(node.inputs().size(), 1);
    EXPECT_EQ(node.inputs()[0], vid);
}

TEST_F(GraphTest, set_node_output_adds_output_to_node) {
    NodeID nid = h.build_node<Operation<float, TestUnaryOp<float>>>();
    INode<float>& node = h.graph.get_node(nid);

    ValueID vid = ValueID{42};
    h.set_node_output(node, vid);

    ASSERT_FALSE(node.outputs().empty());
    EXPECT_EQ(node.outputs().back(), vid);
}

TEST_F(GraphTest, produced_and_consumer_book_keeping_can_be_connected_manually) {
    ValueID input = h.new_input_value();
    NodeID node = h.build_node<Operation<float, TestUnaryOp<float>>>();

    h.append_consumer_table(node, input, 0);

    const std::vector<ConsumerInfo> consumers = h.graph.get_consumed_by(input);
    ASSERT_EQ(consumers.size(), 1);
    EXPECT_EQ(consumers[0].nid, node);
    EXPECT_EQ(consumers[0].in_slot, 0);

    ProducerInfo producer = h.graph.get_produced_by(input);
    EXPECT_EQ(producer.nid, kNoNode);
    EXPECT_EQ(producer.out_slot, 0);
}