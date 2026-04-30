#ifndef FUSION_AUTODIFF_AUTODIFF_CONTEXT_HPP
#define FUSION_AUTODIFF_AUTODIFF_CONTEXT_HPP

#include <vector>

#include "Fusion/common/Checks.hpp"

#include "GradStore.hpp"

template <typename T> class Engine;

template <typename T> class AutodiffContext {
 public:
   static AutodiffRunTime<T> &runtime() {
      static AutodiffRunTime<T> rt{};
      return rt;
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

   static void pop() {
      FUSION_CHECK(!instance_.empty(), "No Engine instance to pop");
      instance_.pop_back();
   }

   static void clear_runtime() noexcept {
      AutodiffRunTime<T>& rt = runtime();
      rt.grad_store().clear_all();
   }

   static void clear() noexcept {
      instance_.clear();
   }

 private:
   inline static thread_local std::vector<Engine<T> *> instance_;
};

template <typename T> struct EngineScope {

   EngineScope() : eng_(AutodiffContext<T>::runtime().grad_store()) {};

   EngineScope(const EngineScope &) = delete;
   EngineScope &operator=(const EngineScope &) = delete;

   EngineScope(EngineScope &&) = delete;
   EngineScope &operator=(EngineScope &&) = delete;

   ~EngineScope() {
      if (active_) {
         exit();
      }
   }

   void enter() {
      AutodiffContext<T>::set(&eng_);
      active_ = true;
   }
   void exit() {
      AutodiffContext<T>::pop();
      active_ = false;
   }

   Engine<T> &eng() { return eng_; }
   bool active() const { return active_; }

 private:
   Engine<T> eng_;
   bool active_{false};
};

#endif // FUSION_AUTODIFF_AUTODIFF_CONTEXT_HPP
