#ifndef MULTIPLY_H
#define MULTIPLY_H

#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"


template <typename T>
struct Multiply {
    inline static constexpr std::string_view name = "Multiply";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context<T>& context, const In& input) {
        std::cout << input.size() << std::endl;
        FUSION_CHECK(input.size() >= 2, "Multiply::forward requires two inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        FUSION_BOUNDS_CHECK(1, input.size());
    	const auto& a = input[0];
    	const auto& b = input[1];
        FUSION_CHECK(a.size() == b.size(), "Multiply::forward input size mismatch");
        context.save("a", input[0]);
        context.save("b", input[1]);
        Tensor<T> c = a * b;
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Multiply::backward expects exactly 1 upstream grad tensor");
        auto& a = context.template load<Tensor<T>>("a");
        auto& b = context.template load<Tensor<T>>("b");
        Tensor<T> g0 = std::move(grad_out[0]);
        FUSION_CHECK(!g0.empty(), "Multiply::backward: upstream grad is empty");
        Tensor<T> c = g0 * b;
        Tensor<T> d = g0 * a;
        GradIn g;
        g.push_back(std::move(c));
        g.push_back(std::move(d));
        return g;
    }
};

#endif // MULTIPLY_H
