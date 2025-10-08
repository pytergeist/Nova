#ifndef POW_H
#define POW_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"


template <typename T>
struct Pow {
    inline static constexpr std::string_view name = "Pow";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context& context, const In& input) {
        FUSION_CHECK(input.size() >= 2, "Pow requires two inputs");
        FUSION_BOUNDS_CHECK(0, input.size());
        FUSION_BOUNDS_CHECK(1, input.size());
        const auto& a = input.at(0);
        const auto& b = input.at(1);
        FUSION_CHECK(a.size() == b.size(), "Pow: input size mismatch");
        context.save("a", input[0]);
        context.save("b", input[1]);
        std::vector<T> c(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            c[i] = std::pow(a[i], b[i]);
        }
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        if (grad_out.size() == 0) return {};
        FUSION_CHECK(grad_out.size() == 1, "Pow::backward expects exactly 1 upstream grad tensor");
        std::vector<T> a = context.template load<std::vector<T>>("a");
        std::vector<T> b = context.template load<std::vector<T>>("b");
        const auto& g0 = grad_out[0];
        std::vector<T> c(g0.size());
        std::vector<T> d(g0.size());
        FUSION_CHECK(!g0.empty(), "Pow::backward: upstream grad is empty");
        for (size_t i = 0; i < g0.size(); ++i) {
            const T& ai = a[i];
            const T& bi = b[i];
            const T& grad = g0[i];

            c.at(i) = std::pow((bi * ai), (bi -1)) * g0;
            d.at(i) = (std::pow(ai, bi) * std::log(ai)) * g0;
        }
        GradIn g;
        g.push_back(std::move(c));
        g.push_back(std::move(d));
        return g;
    }
};

#endif // POW_H
