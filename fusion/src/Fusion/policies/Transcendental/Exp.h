#ifndef EXP_H
#define EXP_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"


template <typename T>
struct Exp {
    inline static constexpr std::string_view name = "Exp";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context<T>& context, const In& input) {
        FUSION_CHECK(input.size() >= 1, "Exp requires one inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        const auto& a = input[0];
        context.save("c", a.clone());
        Tensor<T> c = a.exp();
		Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context<T>& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Exp::backward expects exactly 1 upstream grad tensor");
        Tensor<T> g0 = std::move(grad_out[0]);
    	FUSION_CHECK(!g0.empty(), "Exp::backward: upstream grad is empty");
        auto& a = context.template load<Tensor<T>>("c");
        Tensor<T> g1 = g0 * a;
        GradIn g;
        g.push_back(std::move(g1));
        return g;
    }
};

#endif // EXP_H
