// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef FUSION_CPU_BACKEND_CONCEPT_HPP
#define FUSION_CPU_BACKEND_CONCEPT_HPP

#include <concepts>

/* TODO: refactor this into multiple concepts */

template <typename B>
concept BackendCore = requires {
   typename B::U;
   typename B::vec;
   typename B::wide_vec;

   { B::kVectorBytes } -> std::convertible_to<std::size_t>;
   { B::kLanes } -> std::convertible_to<std::size_t>;
   { B::kUnroll } -> std::convertible_to<std::size_t>;
   { B::kBlock } -> std::convertible_to<std::size_t>;

   { B::kStepVec } -> std::convertible_to<std::size_t>;
   { B::kStep } -> std::convertible_to<std::size_t>;
};

template <typename B>
concept BackendLoadStore =
    requires(const typename B::U *ptr, typename B::U *out, typename B::vec v,
             typename B::wide_vec wv) {
       { B::wide_load(ptr) } -> std::same_as<typename B::wide_vec>;
       { B::load(ptr) } -> std::same_as<typename B::vec>;

       { B::wide_store(out, wv) };
       { B::store(out, v) };
    };

template <typename B>
concept BackendComparison = requires(typename B::vec v, typename B::mask m) {
   { B::cgt(v, v) } -> std::same_as<typename B::mask>;
   { B::cge(v, v) } -> std::same_as<typename B::mask>;
   {
      B::duplicate(std::declval<typename B::U>())
   } -> std::same_as<typename B::vec>;
   { B::blend(m, v, v) } -> std::same_as<typename B::vec>;
};

template <typename B>
concept BackendArithmetic =
    requires(typename B::vec v, typename B::wide_vec wv) {
       { B::add(v, v) } -> std::same_as<typename B::vec>;
       { B::sub(v, v) } -> std::same_as<typename B::vec>;
       { B::mul(v, v) } -> std::same_as<typename B::vec>;
       { B::div(v, v) } -> std::same_as<typename B::vec>;

       { B::maximum(v, v) } -> std::same_as<typename B::vec>;
       { B::pow(v, v) } -> std::same_as<typename B::vec>;
    };

template <typename B>
concept BackendTranscendental = requires(typename B::vec v) {
   { B::sqrt(v) } -> std::same_as<typename B::vec>;
   { B::log(v) } -> std::same_as<typename B::vec>;
   { B::exp(v) } -> std::same_as<typename B::vec>;
};

template <typename B>
concept BackendReduction = requires(typename B::vec v) {
   { B::horizontal_add(v) } -> std::same_as<typename B::U>;
};

template <typename B>
concept BackendConcept =
    BackendCore<B> && BackendLoadStore<B> && BackendComparison<B> &&
    BackendArithmetic<B> && BackendTranscendental<B> && BackendReduction<B>;

#endif // FUSION_CPU_BACKEND_CONCEPT_HPP
