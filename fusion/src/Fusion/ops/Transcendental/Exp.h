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
      	std::vector<T> c(input.at(0).size());
        for (size_t i = 0; i < input.at(0).size(); ++i) {
            c[i] = std::exp(input.at(0).at(i));
        }
        context.save("c", c);
		Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        std::vector<T> d(grad_out.at(0).size());
        std::vector<T> a = context.template load<std::vector<T>>("c");
        for (size_t i = 0; i < grad_out.at(0).size(); ++i) {
            const T& ai = a[i];
            const T& dyi = grad_out.at(0).at(i);
            d[i] = dyi * ai;
        }
        GradIn g;
        g.push_back(std::move(d));
        return g;
    }
};

#endif // EXP_H
