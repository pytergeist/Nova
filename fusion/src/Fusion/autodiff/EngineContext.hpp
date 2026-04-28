#ifndef ENGINE_CONTEXT_HPP
#define ENGINE_CONTEXT_HPP

#include <vector>

#include "Fusion/common/Checks.hpp"

template <typename T> class Engine;

template <typename T> class EngineContext {
 public:
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

   static void clear() noexcept {
      instance_.clear();
   }

 private:
   inline static thread_local std::vector<Engine<T> *> instance_;
};

template <typename T> struct EngineScope {

   EngineScope() = default;

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
      EngineContext<T>::set(&eng_);
      active_ = true;
   }
   void exit() {
      EngineContext<T>::pop();
      active_ = false;
   }

   Engine<T> &eng() { return eng_; }
   bool active() const { return active_; }

 private:
   Engine<T> eng_;
   bool active_{false};
};

#endif // ENGINE_CONTEXT_HPP
