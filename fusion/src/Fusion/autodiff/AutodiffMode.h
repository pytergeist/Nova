#ifndef AUTODIFF_MODE_H
#define AUTODIFF_MODE_H


namespace autodiff {
    inline thread_local bool g_enable_grad = true;

    struct NoGradGuard {
        bool prev_;
        NoGradGuard() : prev_(g_enable_grad) { g_enable_grad = false; }
        ~NoGradGuard() { g_enable_grad = prev_; }
    };

    inline bool grad_enabled() { return g_enable_grad; }
}

#endif // AUTODIFF_MODE_H
