#ifndef ENGINE_CONTEXT_H
#define ENGINE_CONTEXT_H

#include <iostream>


#include "../common/Checks.h"


template <typename T> class Engine;

template <typename T>
class EngineContext {
public:
    static Engine<T>& get() {
        FUSION_CHECK(instance_ != nullptr, "No Engine instance set in context");
        return *instance_;
    }

    static void set(Engine<T>* engine) {
        instance_ = engine;
    }

    static bool has_instance() noexcept { return instance_ != nullptr; }

private:
    inline static thread_local Engine<T>* instance_ = nullptr;
};

#endif // ENGINE_CONTEXT_H
