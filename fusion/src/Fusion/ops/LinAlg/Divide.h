#ifndef DIVIDE_H
#define DIVIDE_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"

template <typename T>
struct Divide {
    inline static constexpr std::string_view name = "Divide";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context& context, const In& input) {
        FUSION_CHECK(input.size() >= 2, "Divide requires two inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        FUSION_BOUNDS_CHECK(1, input.size());
        context.save("a", input[0]);
        context.save("b", input[1]);
        const auto& a = input[0];
    	const auto& b = input[1];
        FUSION_CHECK(a.size() == b.size(), "Divide: input size mismatch");
        std::vector<T> c(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            c[i] = a[i] / b[i];
        }
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Divide::backward expects exactly 1 upstream grad tensor");
        std::vector<T> a = context.template load<std::vector<T>>("a");
        std::vector<T> b = context.template load<std::vector<T>>("b");
         const auto& g0 = grad_out[0];
        std::vector<T> c(g0.size());
        std::vector<T> d(g0.size());
        FUSION_CHECK(!g0.empty(), "Divide::backward: upstream grad is empty");
        for (size_t i = 0; i < g0.size(); ++i) {
            const T& ai = a[i];
            const T& bi = b[i];
            const T& dyi = g0[i];

            c[i] = dyi / bi;
            d[i] = -dyi * ai / (bi * bi);
        }
        GradIn g;
        g.push_back(std::move(c));
        g.push_back(std::move(d));
        return g;
    }
};

#endif // DIVIDE_H
