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


// AutodiffBridge.h
#pragma once
#include "AutodiffMode.h"
#include "EngineContext.h"
#include "Engine.h"

template <typename T>
inline void set_autodiff_enabled(bool on) {
  autodiff::g_enable_grad = on;

  static thread_local Engine<T> kDefaultEngine;
  EngineContext<T>::set(on ? &kDefaultEngine : nullptr);
}

#endif // ENGINE_CONTEXT_H
