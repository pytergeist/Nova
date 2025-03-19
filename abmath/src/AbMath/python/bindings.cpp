#include "../templates/operations.h"
#include "../templates/vector.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(abmath, m) {
  m.doc() = "abmath linear algebra module";

  // Expose the Vector class for doubles (you can similarly expose for other
  // types)
  py::class_<abmath::Vector<double>>(m, "Vector")
      .def(py::init<std::size_t>())
      .def(py::init<const std::vector<double> &>()) // Constructor from
                                                    // std::vector<double>
      .def("__getitem__",
           [](const abmath::Vector<double> &self, std::size_t i) {
             if (i >= self.size())
               throw py::index_error();
             return self[i];
           })
      .def("__setitem__",
           [](abmath::Vector<double> &self, std::size_t i, double value) {
             if (i >= self.size())
               throw py::index_error();
             self[i] = value;
           })
      .def("size", &abmath::Vector<double>::size)
      .def("print", &abmath::Vector<double>::print)
      // New method to convert to a NumPy array.
      .def("to_numpy",
           [](const abmath::Vector<double> &self) {
             // Create a NumPy array with the same size as the vector.
             auto result = py::array_t<double>(self.size());
             // Request a buffer info from the array.
             auto buf = result.request();
             double *ptr = static_cast<double *>(buf.ptr);
             // Copy data from the vector into the NumPy array.
             for (std::size_t i = 0; i < self.size(); i++) {
               ptr[i] = self[i];
             }
             return result;
           })
      .def("__repr__", [](const abmath::Vector<double> &self) {
        std::string s = "Vector([";
        for (std::size_t i = 0; i < self.size(); ++i) {
          s += std::to_string(self[i]);
          if (i != self.size() - 1)
            s += ", ";
        }
        s += "])";
        return s;
      });

  // Expose the add operation with overloads
  // Vector + Vector
  m.def(
      "add",
      [](const abmath::Vector<double> &a, const abmath::Vector<double> &b) {
        return a + b;
      },
      "Add two vectors");

  // Vector + scalar
  m.def(
      "add",
      [](const abmath::Vector<double> &a, double scalar) { return a + scalar; },
      "Add a vector and a scalar");

  // Scalar + Vector
  m.def(
      "add",
      [](double scalar, const abmath::Vector<double> &a) { return scalar + a; },
      "Add a scalar and a vector");

  // Scalar + Scalar
  m.def("add", [](double a, double b) { return a + b; }, "Add two scalars");
}
