#ifndef FUSION_TESTS_AUTODIFF_HELPERS_H_
#define FUSION_TESTS_AUTODIFF_HELPERS_H_

#include "Fusion/autodiff/ADTensor.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/Engine.hpp"
#include "Fusion/autodiff/EngineContext.hpp"

inline ADTensor<float> make_test_tensor(bool requires_grad) {
   return ADTensor<float>({2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6},
                          DType::FLOAT32, Device{DeviceType::CPU, 0},
                          requires_grad);
}

struct EngineContextReset {
   inline ~EngineContextReset() { EngineContext<float>::set(nullptr); }
};


struct UnaryTag{};
struct BinaryTag{};
struct SplitTag{};

template <> struct OpTraits<UnaryTag> {
   static constexpr std::string_view name = "TestUnaryOp";
   static constexpr OpSchema schema{
      .category = OpCategory::EwiseUnary,
      .inputs = {.kind=ArityKind::Fixed, .arity=1},
      .outputs = {.kind=ArityKind::Fixed, .arity=1},
      .mutation = MutationKind::OutOfPlace,
  };
};


template <> struct OpTraits<BinaryTag> {
   static constexpr std::string_view name = "TestBinaryOp";
   static constexpr OpSchema schema{
      .category = OpCategory::EwiseBinary,
      .inputs = {.kind=ArityKind::Fixed, .arity=2},
      .outputs = {.kind=ArityKind::Fixed, .arity=1},
      .mutation = MutationKind::OutOfPlace,
  };
};


template <> struct OpTraits<SplitTag> {
   static constexpr std::string_view name = "TestSplitOp";
   static constexpr OpSchema schema{
      .category = OpCategory::EwiseBinary,
      .inputs = {.kind=ArityKind::Fixed, .arity=2},
      .outputs = {.kind=ArityKind::Fixed, .arity=2},
      .mutation = MutationKind::OutOfPlace,
  };
};



template <typename T>
struct TestUnaryOp {
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

template <typename T>
struct TestBinaryOp {
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

template <typename T>
struct TestSplitOp {
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

#endif // FUSION_TESTS_AUTODIFF_HELPERS_H_