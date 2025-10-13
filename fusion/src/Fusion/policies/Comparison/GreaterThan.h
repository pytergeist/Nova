#ifndef GREATER_THAN_H
#define GREATER_THAN_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"
#include "../../TensorFactory.h"

template <typename T>
struct GreaterThan {
    inline static constexpr std::string_view name = "GreaterThan";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context<T>& context, const In& input) {
        FUSION_CHECK(input.size() >= 2, "GreaterThan requires two inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        FUSION_BOUNDS_CHECK(1, input.size());
        const auto& a = input[0];
        const auto& b = input[1];
        context.save("a", a);
        context.save("b", b);
        FUSION_CHECK(a.size() == b.size(), "GreaterThan: input size mismatch");
        Tensor<T> c = a > b;
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "GreaterThan::backward expects exactly 1 upstream grad tensor");
        const Tensor<T>& a = context.template load<Tensor<T>>("a");
        const Tensor<T>& b = context.template load<Tensor<T>>("b");
        const auto& g0 = grad_out[0];
        FUSION_CHECK(!g0.empty(), "GreaterThan::backward: upstream grad is empty");
        Tensor<T> c = zeros_like(a);
        Tensor<T> d = zeros_like(b);
        GradIn g;
        g.push_back(std::move(c));
        g.push_back(std::move(d));
        return g;
    }
};

#endif // GREATER_THAN_H
