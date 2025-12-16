#ifndef AUTODIFF_BRIDGE_HPP
#define AUTODIFF_BRIDGE_HPP

#include "AutodiffMode.hpp"
#include "Engine.hpp"
#include "EngineContext.hpp"

template <typename T> inline void set_autodiff_enabled(bool on) {
   autodiff::g_enable_grad = on;

   static thread_local Engine<T> kDefaultEngine;
   EngineContext<T>::set(on ? &kDefaultEngine : nullptr);
}

#endif // AUTODIFF_BRIDGE_HPP
