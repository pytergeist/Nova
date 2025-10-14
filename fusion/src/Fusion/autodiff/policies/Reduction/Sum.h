#ifndef SUM_H
#define SUM_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"
#include "../../TensorFactory.h"


template <typename T>
struct Sum {
    inline static constexpr std::string_view name = "Exp";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context<T>& context, const In& input) {
        FUSION_CHECK(input.size() >= 1, "Sum requires one inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        const auto& a = input[0];
        Tensor<T> c = a.sum();
        context.save("c", a);
        Out out;
        out.push_back(c);
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Sum::backward expects exactly 1 upstream grad tensor");
        Tensor<T> g0 = std::move(grad_out[0]);
        FUSION_CHECK(!g0.empty(), "Sum::backward: upstream grad is empty");
        Tensor<T> g1;
        if (g0.size() == 1) {
            const Tensor<T>& a = context.template load<Tensor<T>>("c");
            Tensor<T> g1 = ones_like<T>(a) * g0;
        } else {
        Tensor<T> g1 = g0;
        }
        GradIn g;
        g.push_back(g1);
        return g;
    }
};

#endif // SUM_H
