#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include <string_view>
#include <vector>

#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/core/RawTensor.hpp"

template <typename T> struct Transpose {
   static constexpr std::string_view name = "Transpose";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) { // NOLINT
      FUSION_CHECK(!input.empty(), "Transpose requires one inputs");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = input.at(0);
      RawTensor<T> y = x.transpose();
      Out out;
      out.push_back(y);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) { // NOLINT
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(
          grad_out.size() == 1,
          "Transpose::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      RawTensor<T> g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Transpose::backward: upstream grad is empty");
      RawTensor<T> gx = g0.transpose();
      GradIn g;
      g.push_back(gx);
      return g;
   }
};

#endif // TRANSPOSE_HPP
