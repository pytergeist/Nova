#ifndef FUSION_AUTODIFF_GRAD_STORE_HPP
#define FUSION_AUTODIFF_GRAD_STORE_HPP

#include <vector>

#include "ADTypes.h"

#include "Fusion/core/RawTensor.hpp"

struct LeafGradBinding {
   ValueID vid;
   GradSlotID slot;
};

template <typename T>
class GradStore {
public:

   GradStore() = default;
   GradStore(const GradStore &) = delete;
   GradStore &operator=(const GradStore &) = delete;
   GradStore(GradStore &&) noexcept = delete;
   GradStore &operator=(GradStore &&) noexcept = delete;

   GradSlotID allocate() {
      slots_.emplace_back(std::nullopt);
      return slots_.size() - 1;
   }

   bool has(const GradSlotID slot) const noexcept {
      return slot <= slots_.size();
   }
   void set(const GradSlotID slot, const RawTensor<T>& grad) noexcept {
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

template <typename T>
class AutodiffRunTime {
   public:
      GradStore<T>& grad_store () {return grad_store_;}
   private:
      GradStore<T> grad_store_;
};


#endif // FUSION_AUTODIFF_GRAD_STORE_HPP