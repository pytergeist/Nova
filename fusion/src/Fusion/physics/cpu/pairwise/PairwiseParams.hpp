#ifndef FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_PARAMS_H_
#define FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_PARAMS_H_

#include <utility>

#include "PairwiseTags.hpp"

template <typename T> struct LJParams {
   T epsilon;
   T sigma;
};

struct NoParams {};

template <class Tag, class T> struct params_type {
   using type = NoParams;
};

template <class T> struct params_type<LJEnergy, T> {
   using type = LJParams<T>;
};

template <class T> struct params_type<LJForce, T> {
   using type = LJParams<T>;
};

template <class Tag, class T>
using params_type_t = typename params_type<Tag, T>::type;

#endif // FUSION_PHYSICS_CPU_PAIRWISE_PAIRWISE_PARAMS_H_