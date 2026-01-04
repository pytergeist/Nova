// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef MAXIMUM_HPP
#define MAXIMUM_HPP

#include <string_view>
#include <vector>

#include "Fusion/TensorFactory.hpp"
#include "Fusion/autodiff/AutodiffMeta.hpp"
#include "Fusion/autodiff/AutodiffMode.hpp"
#include "Fusion/autodiff/registry/Operation.hpp"
#include "Fusion/common/Checks.hpp"

template <typename T> struct Maximum {
   static constexpr std::string_view name = "Maximum";
   using In = AutodiffMeta<T>;
   using Out = AutodiffMeta<T>;
   using GradIn = AutodiffMeta<T>;
   using GradOut = AutodiffMeta<T>;

   Out forward(Context<T> &context, const In &input) {
      FUSION_CHECK(input.size() >= 2, "Maximum requires two inputs");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = input.at(0);
      const RawTensor<T> &y = input.at(1);
      context.save("x", x);
      context.save("y", y);
      FUSION_ALLOW_SCALAR_BINARY(x, y);
      RawTensor<T> z = x.maximum(y);
      Out out;
      out.push_back(z);
      return out;
   };

   GradIn backward(Context<T> &context, GradOut &grad_out) {
      if (grad_out.empty()) {
         return {};
      }
      FUSION_CHECK(grad_out.size() == 1,
                   "Maximum::backward expects exactly 1 upstream grad tensor");
      const autodiff::NoGradGuard _;
      const RawTensor<T> &x = context.template load<RawTensor<T>>("x");
      const RawTensor<T> &y = context.template load<RawTensor<T>>("y");
      const RawTensor<T> &g0 = grad_out.at(0);
      FUSION_CHECK(!g0.empty(), "Maximum::backward: upstream grad is empty");
      RawTensor<T> gx = g0 * (x >= y);
      RawTensor<T> gy = g0 * (y > x);
      GradIn g;
      g.push_back(gx);
      g.push_back(gy);
      return g;
   }
};

#endif // MAXIMUM_HPP
