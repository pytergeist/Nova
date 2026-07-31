#ifndef FUSION_TESTS_AUTODIFF_HELPERS_H_
#define FUSION_TESTS_AUTODIFF_HELPERS_H_

#include <gtest/gtest.h>
#include <random>

#include "Fusion/autodiff/ADTensor.hpp"
#include "Fusion/autodiff/AutodiffContext.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/Engine.hpp"

template <typename T> class GraphHarness {
 public:
   Graph<T> graph;

   NodeID make_node_id() { return graph.make_node_id(); }

   ValueID new_input_value() { return graph.new_input_value(); }

   ValueID new_intermediate_value() { return graph.new_intermediate_value(); }

   void add_edge(NodeID src_nid, NodeID dst_nid) {
      graph.add_edge(src_nid, dst_nid);
   }

   template <typename ConcreteOp> NodeID build_node() {
      return graph.template build_node<ConcreteOp>();
   }

   void append_consumer_table(NodeID dst_nid, ValueID vid, size_t slot) {
      graph.append_consumer_table(dst_nid, vid, slot);
   }

   void set_produced_by(ValueID vid, NodeID nid, size_t out_slot) {
      graph.set_produced_by(vid, nid, out_slot);
   }

   void set_node_input(INode<T> &node, ValueID vid) {
      graph.set_node_input(node, vid);
   }

   void set_node_output(INode<T> &node, ValueID vid) {
      graph.set_node_output(node, vid);
   }

   const auto &edges() const { return graph.edges_; }
};

class GraphTest : public ::testing::Test {
 protected:
   using T = float;
   GraphHarness<T> h;
};

inline std::vector<float> generate_random_vector(std::vector<std::size_t> shape,
                                                 std::uint32_t seed,
                                                 float min = 0.0,
                                                 float max = 10.0) {
   std::mt19937 engine_;
   engine_.seed(seed);
   size_t total = std::accumulate(shape.begin(), shape.end(),
                                  static_cast<size_t>(1), std::multiplies<>());

   std::vector<float> data;
   data.reserve(total);

   std::uniform_real_distribution<float> dist(min, max);
   for (size_t i = 0; i < total; ++i) {
      data.push_back(dist(engine_));
   }
   return data;
}

inline DenseTensor<float> make_test_raw_tensor(
    uint32_t seed = 42,
    std::vector<std::size_t> shape = std::vector<size_t>{2, 3}) {
   std::vector<float> data = generate_random_vector(shape, seed);
   return DenseTensor<float>(shape, data, DType::FLOAT32,
                             Device{DeviceType::CPU, 0});
}

inline Tensor<float>
make_test_tensor(uint32_t seed = 42,
                 std::vector<std::size_t> shape = std::vector<size_t>{2, 3}) {
   DenseTensor<float> dense = make_test_raw_tensor(seed, shape);
   return Tensor<float>::from_dense(dense);
}

inline ADTensor<float>
make_test_tensor(bool requires_grad, uint32_t seed = 42,
                 std::vector<std::size_t> shape = std::vector<size_t>{2, 3}) {
   std::vector<float> data = generate_random_vector(shape, seed);
   return ADTensor<float>(shape, data, DType::FLOAT32,
                          Device{DeviceType::CPU, 0}, requires_grad);
}

inline AutodiffMeta<float> make_test_meta_forward_result(bool requires_grad,
                                                         int num_tensors,
                                                         uint32_t seed) {
   AutodiffMeta<float> meta;
   for (int i = 0; i < num_tensors; ++i) {
      meta.push_back(make_test_tensor(requires_grad, seed).base());
      seed += 1;
   }
   return meta;
}

inline void EXPECT_RAW_TENSOR_EQ(const DenseTensor<float> &actual,
                                 const DenseTensor<float> &expected) {
   EXPECT_EQ(actual.shape(), expected.shape());
   EXPECT_EQ(actual.dtype(), expected.dtype());
   EXPECT_EQ(actual.device(), expected.device());
   EXPECT_EQ(actual.size(), expected.size());
   EXPECT_EQ(actual.is_contiguous(), expected.is_contiguous());

   for (size_t i = 0; i < actual.size(); ++i) {
      EXPECT_FLOAT_EQ(actual[i], expected[i]);
   }
}

inline void EXPECT_TENSOR_EQ(const DenseTensor<float> &actual,
                             const DenseTensor<float> &expected) {
   EXPECT_RAW_TENSOR_EQ(actual, expected);
}

inline void EXPECT_TENSOR_EQ(const Tensor<float> &actual,
                             const DenseTensor<float> &expected) {
   ASSERT_TRUE(actual.is_dense());
   EXPECT_RAW_TENSOR_EQ(actual.dense(), expected);
}

inline void EXPECT_TENSOR_EQ(const DenseTensor<float> &actual,
                             const Tensor<float> &expected) {
   ASSERT_TRUE(expected.is_dense());
   EXPECT_RAW_TENSOR_EQ(actual, expected.dense());
}

inline void EXPECT_TENSOR_EQ(const Tensor<float> &actual,
                             const Tensor<float> &expected) {
   ASSERT_EQ(actual.layout(), expected.layout());

   if (actual.is_dense()) {
      EXPECT_RAW_TENSOR_EQ(actual.dense(), expected.dense());
      return;
   }

   EXPECT_EQ(actual.logical_shape(), expected.logical_shape());
   EXPECT_EQ(actual.storage_shape(), expected.storage_shape());
   EXPECT_RAW_TENSOR_EQ(actual.physical_base(), expected.physical_base());
}

struct AutodiffContextReset {
   // TODO: this can be made a feature with setup/teardown -- cleaner
   ~AutodiffContextReset() {
      AutodiffContext<float>::clear();
      AutodiffContext<float>::clear_runtime();
   }
};

struct UnaryTag {};
struct BinaryTag {};
struct SplitTag {};

template <> struct OpTraits<UnaryTag> {
   static constexpr std::string_view name = "TestUnaryOp";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseUnary,
       .inputs = {.kind = ArityKind::Fixed, .arity = 1},
       .outputs = {.kind = ArityKind::Fixed, .arity = 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<BinaryTag> {
   static constexpr std::string_view name = "TestBinaryOp";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {.kind = ArityKind::Fixed, .arity = 2},
       .outputs = {.kind = ArityKind::Fixed, .arity = 1},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <> struct OpTraits<SplitTag> {
   static constexpr std::string_view name = "TestSplitOp";
   static constexpr OpSchema schema{
       .category = OpCategory::EwiseBinary,
       .inputs = {.kind = ArityKind::Fixed, .arity = 1},
       .outputs = {.kind = ArityKind::Fixed, .arity = 2},
       .mutation = MutationKind::OutOfPlace,
   };
};

template <typename T> struct TestUnaryOp {
   using tag = UnaryTag;
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      Out out;
      return out;
   }

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      GradIn g;
      return g;
   }
};

template <typename T> struct TestBinaryOp {
   using tag = BinaryTag;
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      Out out;
      return out;
   }

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      GradIn g;
      return g;
   }
};

template <typename T> struct TestSplitOp {
   using tag = SplitTag;
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      Out out;
      return out;
   }

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      GradIn g;
      return g;
   }
};

/// The below test Op fixtures should be used when test require
/// non-empty forward/backward result

template <typename T, std::uint32_t SEED = 42> struct PopTestUnaryOp {
   std::uint32_t seed = SEED;
   using tag = UnaryTag;
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      Out out = make_test_meta_forward_result(false, 1, seed);
      return out;
   }

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      GradIn g = make_test_meta_forward_result(false, 1, seed);
      return g;
   }
};

template <typename T, std::uint32_t SEED = 42> struct PopTestBinaryOp {
   std::uint32_t seed = SEED;
   using tag = BinaryTag;
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      Out out = make_test_meta_forward_result(false, 1, seed);
      return out;
   }

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      GradIn g = make_test_meta_forward_result(false, 2, seed);
      return g;
   }
};

template <typename T, std::uint32_t SEED = 42> struct PopTestSplitOp {
   std::uint32_t seed = SEED;
   using tag = SplitTag;
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      Out out = make_test_meta_forward_result(false, 2, seed);
      return out;
   }

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      GradIn g = make_test_meta_forward_result(false, 2, seed);
      return g;
   }
};

#endif // FUSION_TESTS_AUTODIFF_HELPERS_H_