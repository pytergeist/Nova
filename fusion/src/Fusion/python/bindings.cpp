#include "../tensor/tensor.h"
#include <algorithm> // for std::copy
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

PYBIND11_MODULE(fusion_math, m) {
  m.doc() = "Fusion Tensor module exposing Tensor<double>";

  py::class_<Tensor<double>>(m, "Tensor")
      // Construct from shape + list of values.
      .def(py::init([](const std::vector<size_t> &shape,
                       const std::vector<double> &values) -> Tensor<double> {
             if (shape.size() != 2) {
               throw std::runtime_error(
                   "Shape must be a vector of 2 elements [rows, cols]");
             }
             size_t rows = shape[0], cols = shape[1];
             if (values.size() != rows * cols) {
               throw std::runtime_error(
                   "The number of values does not match the provided shape");
             }
             Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>
                 mat(rows, cols);
             for (size_t i = 0; i < rows; ++i) {
               for (size_t j = 0; j < cols; ++j) {
                 mat(i, j) = values[i * cols + j];
               }
             }
             return Tensor<double>(mat);
           }),
           "Construct a Tensor from shape and a list of values.")

      // Construct from scalar.
      .def(py::init([](const double &v) -> Tensor<double> {
             Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>
                 mat(1, 1);
             mat(0, 0) = v;
             return Tensor<double>(mat);
           }),
           "Construct a Tensor from a scalar.")

      .def("__repr__",
           [](const Tensor<double> &t) {
             std::ostringstream oss;
             oss << t;
             return oss.str();
           })

      // to_numpy now squeezes length‑1 dimensions.
      .def(
          "to_numpy",
          [](const Tensor<double> &t) {
            // access the Eigen storage
            auto cpuStorage = dynamic_cast<const EigenTensorStorage<double> *>(
                t.storage.get());
            if (!cpuStorage) {
              throw std::runtime_error(
                  "Tensor does not use EigenTensorStorage");
            }

            auto rows = cpuStorage->rows();
            auto cols = cpuStorage->cols();

            // choose a 1D shape if one dimension is 1
            std::vector<py::ssize_t> shape;
            std::vector<py::ssize_t> strides;
            if (cols == 1 && rows > 1) {
              // column vector -> (rows,)
              shape = {py::ssize_t(rows)};
              strides = {sizeof(double)};
            } else if (rows == 1 && cols > 1) {
              // row vector -> (cols,)
              shape = {py::ssize_t(cols)};
              strides = {sizeof(double)};
            } else {
              // full matrix
              shape = {py::ssize_t(rows), py::ssize_t(cols)};
              strides = std::vector<py::ssize_t>(2);
              // row-major
              strides[1] = sizeof(double);
              strides[0] = strides[1] * cols;
            }

            // create the array
            py::array_t<double> np_arr(shape, strides);
            auto buf = np_arr.request();
            double *ptr = static_cast<double *>(buf.ptr);
            size_t numElements = rows * cols;
            std::copy(cpuStorage->matrix.data(),
                      cpuStorage->matrix.data() + numElements, ptr);
            return np_arr;
          },
          "Return the tensor as a numpy array (dimensions of size 1 are "
          "squeezed).")

      // operator overloads
      .def("__add__", [](const Tensor<double> &a,
                         const Tensor<double> &b) { return a + b; })
      .def("__sub__", [](const Tensor<double> &a,
                         const Tensor<double> &b) { return a - b; })
      .def("__mul__", [](const Tensor<double> &a,
                         const Tensor<double> &b) { return a * b; })
      .def("__truediv__", [](const Tensor<double> &a,
                             const Tensor<double> &b) { return a / b; })

      // matrix ops & elementwise fns
      .def("__matmul__", &Tensor<double>::matmul,
           "Matrix multiplication of two Tensors.")
      .def("pow",
           py::overload_cast<const double>(&Tensor<double>::pow, py::const_),
           "Raise each element to a scalar power.")
      .def("pow",
           py::overload_cast<const Tensor<double> &>(&Tensor<double>::pow,
                                                     py::const_),
           "Raise each element to the elementwise power given by another "
           "Tensor.")
      .def("sqrt", &Tensor<double>::sqrt, "Element-wise square root.")
      .def("exp", &Tensor<double>::exp, "Element-wise exponential.")
      .def("log", &Tensor<double>::log, "Element-wise natural logarithm.")
      .def("sum", &Tensor<double>::sum, "Sum of all elements.")
      .def("transpose", &Tensor<double>::transpose, "Transpose of the Tensor.")
      .def("maximum", &Tensor<double>::maximum,
           "Element-wise maximum with another Tensor.");
}
