#ifndef FUSION_CPU_BACKEND_CONCEPT_HPP
#define FUSION_CPU_BACKEND_CONCEPT_HPP

#include <concepts>

/* TODO: refactor this into multiple concepts */

template <typename B>
concept BackendConcept =
    requires(const typename B::U *ptr, typename B::U *out, typename B::vec v,
             typename B::wide_vec wv, typename B::mask m) {
       typename B::U;
       typename B::vec;
       typename B::wide_vec;

       { B::kVectorBytes } -> std::convertible_to<std::size_t>;
       { B::kLanes } -> std::convertible_to<std::size_t>;
       { B::kUnroll } -> std::convertible_to<std::size_t>;
       { B::kBlock } -> std::convertible_to<std::size_t>;

       { B::kStepVec } -> std::convertible_to<std::size_t>;
       { B::kStep } -> std::convertible_to<std::size_t>;

       { B::wide_load(ptr) } -> std::same_as<typename B::wide_vec>;
       { B::load(ptr) } -> std::same_as<typename B::vec>;

       { B::wide_store(out, wv) };
       { B::store(out, v) };

       { B::cgt(v, v) } -> std::same_as<typename B::mask>;
       { B::cge(v, v) } -> std::same_as<typename B::mask>;
       {
          B::duplicate(std::declval<typename B::U>())
       } -> std::same_as<typename B::vec>;

       { B::blend(m, v, v) } -> std::same_as<typename B::vec>;

       { B::add(v, v) } -> std::same_as<typename B::vec>;
       { B::sub(v, v) } -> std::same_as<typename B::vec>;
       { B::mul(v, v) } -> std::same_as<typename B::vec>;
       { B::div(v, v) } -> std::same_as<typename B::vec>;

       { B::maximum(v, v) } -> std::same_as<typename B::vec>;
       { B::pow(v, v) } -> std::same_as<typename B::vec>;

       { B::sqrt(v) } -> std::same_as<typename B::vec>;
       { B::log(v) } -> std::same_as<typename B::vec>;
       { B::exp(v) } -> std::same_as<typename B::vec>;
    };

#endif // FUSION_CPU_BACKEND_CONCEPT_HPP
