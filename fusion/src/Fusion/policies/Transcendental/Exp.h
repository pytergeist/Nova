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

    Out forward(Context& context, const In& input) {
        FUSION_CHECK(input.size() >= 1, "Exp requires one inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        const auto& a = input[0];
      	std::vector<T> c(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            c[i] = std::exp(a[i]);
        }
        context.save("c", c);
		Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Exp::backward expects exactly 1 upstream grad tensor");
        const auto& g0 = grad_out[0];
        std::vector<T> g1(g0.size());
    	FUSION_CHECK(!g0.empty(), "Exp::backward: upstream grad is empty");
        std::vector<T> a = context.template load<std::vector<T>>("c");
        for (size_t i = 0; i < g0.size(); ++i) {
            const T& ai = a[i];
            const T& dyi = g0[i];
            g1[i] = dyi * ai;
        }
        GradIn g;
        g.push_back(std::move(g1));
        return g;
    }
};

#endif // EXP_H
