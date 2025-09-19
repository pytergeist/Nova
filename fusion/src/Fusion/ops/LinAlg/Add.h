#ifndef ADD_H
#define ADD_H

#include <vector>
#include "../Operation.h"


template <typename T>
struct Add {
    static constexpr std::string_view name = "Add";
    using In = BinaryType<T>;
    using Out = UnaryType<T>;
    using GradIn = BinaryType<T>;
    using GradOut = UnaryType<T>;

    Out forward(Context& context, const In& input) {
        Out out;
        out.a.resize(input.a.size());
        for (size_t i = 0; i < input.a.size(); ++i) {
            out.a[i] = (input.a[i] + input.b[i]);
        }
        return out;
    };

    GradIn backward(Context& context, GradOut& grad_out) {
        GradIn g;
        g.a = grad_out.a;
        g.b = grad_out.a;
        return g;
    }
};

#endif // ADD_H
