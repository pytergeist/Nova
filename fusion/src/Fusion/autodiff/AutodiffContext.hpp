#ifndef FUSION_AUTODIFF_AUTODIFF_CONTEXT_HPP
#define FUSION_AUTODIFF_AUTODIFF_CONTEXT_HPP

#include <vector>

#include "Fusion/common/Checks.hpp"

#include "GradStore.hpp"

template <typename T> class Engine;

template <typename T> class AutodiffContext {
 public:
   static AutodiffRunTime<T> &runtime() noexcept {
      // Intentionally immortal: tensors need to query gradients after
      // EngineScope exits. Do not destroy during python/cpp static teardown.
      // Previous version was stored in static memory but suffered from
      // static destruction order fiasco. Changed to heap allocated immortal
      // object, this means it is intentionally cleaned up by the OS on
      // programme shutdown. Current design allows for a single type homogenous
      // runtime, this design will need to be expanded in the future to be an
      // immortal runtime register, that manages multiple runtimes (could then
      // use dtype promotion for mixed types). Another random thought on mixed
      // dtypes would be type erased runtimes - food for thought.
      static auto *rt = new AutodiffRunTime<T>{};
      return *rt;
   }

   static Engine<T> &get() {
      FUSION_CHECK(!instance_.empty(), "No Engine instance set in context");
      return *instance_.back();
   }

   static bool has() { return !instance_.empty(); }

   static void set(Engine<T> *engine) {
      FUSION_CHECK(engine, "Trying to set nullptr as engine");
      instance_.push_back(engine);
   }

   static void pop_noexcept() noexcept {
      if (!instance_.empty()) {
         instance_.pop_back();
      }
   }

   static void clear_runtime() noexcept { runtime().clear(); }

   static void clear() noexcept { instance_.clear(); }

 private:
   inline static thread_local std::vector<Engine<T> *> instance_;
};

template <typename T> struct EngineScope {

   EngineScope() : eng_(AutodiffContext<T>::runtime().grad_store()) {};

   EngineScope(const EngineScope &) = delete;
   EngineScope &operator=(const EngineScope &) = delete;

   EngineScope(EngineScope &&) = delete;
   EngineScope &operator=(EngineScope &&) = delete;

   ~EngineScope() noexcept {
      if (active_) {
         try {
            exit();
         } catch (...) {
            std::terminate();
         }
      }
   }

   void enter() {
      AutodiffContext<T>::set(&eng_);
      active_ = true;
   }
   void exit() {
      AutodiffContext<T>::pop_noexcept();
      active_ = false;
   }

   Engine<T> &eng() { return eng_; }
   bool active() const { return active_; }

 private:
   Engine<T> eng_;
   bool active_{false};
};

#endif // FUSION_AUTODIFF_AUTODIFF_CONTEXT_HPP
