#ifndef AUTODIFF_BRIDGE_H
#define AUTODIFF_BRIDGE_H


#include "AutodiffMode.h"
#include "Engine.h"
#include "EngineContext.h"

template <typename T> inline void set_autodiff_enabled(bool on) {
   autodiff::g_enable_grad = on;

   static thread_local Engine<T> kDefaultEngine;
   EngineContext<T>::set(on ? &kDefaultEngine : nullptr);
}

#endif // AUTODIFF_BRIDGE_H
