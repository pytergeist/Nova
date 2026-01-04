// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Tom Pope
//
// Nova — a high-performance hybrid physics and deep learning tensor engine.

#ifndef BIND_RANDOM_HPP
#define BIND_RANDOM_HPP

#include <pybind11/pybind11.h>
#include <string>
#include <typeinfo>

#include "Fusion/Random.hpp"

namespace py = pybind11;

template <typename T> void bind_random(py::module_ &m_ten, const char *name) {
   using ADT = ADTensor<T>;
   using Base = RawTensor<T>;
   using RNG = Random<T>;

   py::class_<RNG>(m_ten, name)
       .def(py::init<uint32_t>(), py::arg("seed") = std::random_device{}())
       .def(
           "uniform_cpp",
           [](RNG &self, const std::vector<size_t> &shape, T min, T max,
              Device device) {
              Base b = self.uniform_base(shape, min, max, device);
              return ADT(std::move(b), false);
           },
           py::arg("shape"), py::arg("min"), py::arg("max"), py::arg("device"));
}

#endif // BIND_RANDOM_HPP
