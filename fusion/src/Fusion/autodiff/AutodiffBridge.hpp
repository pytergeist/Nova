#ifndef AUTODIFF_BRIDGE_HPP
#define AUTODIFF_BRIDGE_HPP

#include "AutodiffContext.hpp"
#include "AutodiffMode.hpp"
#include "Engine.hpp"

template <typename T> void set_autodiff_enabled(bool on) {
   autodiff::g_enable_grad = on;

   thread_local Engine<T> kDefaultEngine(
       AutodiffContext<T>::runtime().grad_store());
   AutodiffContext<T>::set(on ? &kDefaultEngine : nullptr);
}

#endif // AUTODIFF_BRIDGE_HPP
