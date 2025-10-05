#ifndef ADD_H
#define ADD_H

#include <vector>
#include <string_view>
#include "../../autodiff/Traits.h"
#include "../Operation.h"


template <typename T>
struct Add {
    inline static constexpr std::string_view name = "Add";
    using In = MultiTensor<T>;
    using Out = MultiTensor<T>;
    using GradIn = MultiTensor<T>;
    using GradOut = MultiTensor<T>;

    Out forward(Context& context, const In& input) {
        if (input.size() < 2) throw std::runtime_error("Add needs 2 inputs");
    	const auto& a = input.at(0);
    	const auto& b = input.at(1);
    	if (a.size() != b.size()) throw std::runtime_error("Add: size mismatch");
        std::vector<T> c(a.size());
        for (size_t i = 0; i < input.at(0).size(); ++i) {
            c.at(i) = (input.at(0).at(i) + input.at(1).at(i));
        }
        Out out;
        out.push_back(std::move(c));
        return out;
    };

    GradIn backward(Context& context, const GradOut& grad_out) {
        GradIn g;
        g.push_back(std::move(grad_out.at(0)));
        g.push_back(std::move(grad_out.at(0)));
        return g;
    }
};

#endif // ADD_H
