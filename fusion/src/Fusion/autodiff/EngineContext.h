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

    static bool has() {
      return instance_ != nullptr;
    }

    static void set(Engine<T>* engine) {
        instance_ = engine;
    }

    static bool has_instance() noexcept { return instance_ != nullptr; }

private:
    inline static thread_local Engine<T>* instance_ = nullptr;
};


template <typename T>
struct EngineScope {
  Engine<T> eng_;
  bool active_{false};

  EngineScope() = default;
  ~EngineScope() { if (active_) exit(); }

  EngineScope(const EngineScope&) = delete;
  EngineScope& operator=(const EngineScope&) = delete;
  EngineScope(EngineScope&&) = delete;
  EngineScope& operator=(EngineScope&&) = delete;

  void enter() { EngineContext<T>::set(&eng_); active_ = true; }
  void exit()  { EngineContext<T>::set(nullptr); active_ = false; }
};


#endif // ENGINE_CONTEXT_H
