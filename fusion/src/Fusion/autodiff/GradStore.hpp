#ifndef FUSION_AUTODIFF_GRAD_STORE_HPP
#define FUSION_AUTODIFF_GRAD_STORE_HPP

#include <vector>

#include "ADTypes.h"

#include "Fusion/core/RawTensor.hpp"

struct LeafGradBinding {
   ValueID vid;
   GradSlotID slot;

   bool operator==(const LeafGradBinding& other) const noexcept {
      return vid == other.vid && slot == other.slot;
   }
};

struct LeafGradBindingHash {
   std::size_t operator()(const LeafGradBinding& b) const noexcept {
      std::size_t h1 = std::hash<ValueID>{}(b.vid);
      std::size_t h2 = std::hash<GradSlotID>{}(b.slot);

      return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
   }
};

template <typename T> class GradStore {
 public:
   GradStore() = default;
   GradStore(const GradStore &) = delete;
   GradStore &operator=(const GradStore &) = delete;
   GradStore(GradStore &&) noexcept = delete;
   GradStore &operator=(GradStore &&) noexcept = delete;

   GradSlotID allocate() {
      slots_.emplace_back(std::nullopt);
      return static_cast<GradSlotID>(slots_.size() - 1);
   }

   RawTensor<T> get(const GradSlotID slot) {
      FUSION_CHECK(has(slot), "No gradient available in GradStore");
      return slots_[static_cast<std::size_t>(slot)].value();
   }

   bool has(const GradSlotID slot) const noexcept {
      return slot <= slots_.size();
   }
   void set(const GradSlotID slot, const RawTensor<T> &grad) {
      FUSION_BOUNDS_CHECK(slot, slots_.size());
      slots_[slot] = grad;
   }

   void clear(const GradSlotID slot) {
      FUSION_BOUNDS_CHECK(slot, slots_.size());
      slots_[slot].reset();
   }

 private:
   std::vector<std::optional<RawTensor<T>>> slots_{};
};

template <typename T> class AutodiffRunTime {
 public:
   GradStore<T> &grad_store() { return grad_store_; }

 private:
   GradStore<T> grad_store_;
};

#endif // FUSION_AUTODIFF_GRAD_STORE_HPP