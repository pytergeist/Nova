#ifndef SQRT_H
#define SQRT_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"
#include "../../AutodiffMode.h"


template <typename T>
struct Sqrt {
    inline static constexpr std::string_view name = "Sqrt";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context<T>& context, const In& input) {
        FUSION_CHECK(input.size() >= 1, "Sqrt requires one inputs");
        autodiff::NoGradGuard _;
        FUSION_BOUNDS_CHECK(0, input.size());
        const Tensor<T>& a = input[0];
        context.save("c", a);
        Tensor<T> c = a.sqrt();
        Out out;
        out.push_back(c);
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Sqrt::backward expects exactly 1 upstream grad tensor");
        autodiff::NoGradGuard _;
        const auto& g0 = grad_out[0];
        FUSION_CHECK(!g0.empty(), "Sqrt::backward: upstream grad is empty");
        const Tensor<T>& a = context.template load<Tensor<T>>("c");
        Tensor<T> g1 = g0 / (a.sqrt()); // TODO: this is wrong - need 2* (need tensor * scalar)
        GradIn g;
        g.push_back(g1);
        return g;
    }
};

#endif // SQRT_H
