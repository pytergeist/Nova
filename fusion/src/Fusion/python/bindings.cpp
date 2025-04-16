#include "../tensor/tensor.h"
#include <eigen3/Eigen/src/Core/Matrix.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace py = pybind11;

PYBIND11_MODULE(fusion_math, m) {
  m.doc() = "Fusion Tensor module exposing Tensor<double>";

  py::class_<Tensor<double>>(m, "Tensor")
      // Create a Tensor from shape and a list of doubles.
      .def(py::init([](const std::vector<size_t> &shape,
                       const std::vector<double> &values) -> Tensor<double> {
             if (shape.size() != 2) {
               throw std::runtime_error(
                   "Shape must be a vector of 2 elements [rows, cols]");
             }
             size_t rows = shape[0];
             size_t cols = shape[1];
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

      // Create a Tensor from a scalar (creates a 1x1 Tensor).
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

      // Return the tensor as a numpy array.
      .def(
          "to_numpy",
          [](const Tensor<double> &t) {
            // Get the EigenTensorStorage from the Tensor
            auto cpuStorage = dynamic_cast<const EigenTensorStorage<double> *>(
                t.storage.get());
            if (!cpuStorage) {
              throw std::runtime_error(
                  "Tensor does not use EigenTensorStorage");
            }
            // Determine the tensor's shape using the storage's rows and cols.
            std::vector<py::ssize_t> shape = {
                static_cast<py::ssize_t>(cpuStorage->rows()),
                static_cast<py::ssize_t>(cpuStorage->cols())};
            // Compute strides for row-major order.
            std::vector<py::ssize_t> strides(shape.size());
            py::ssize_t stride = sizeof(double);
            for (ssize_t i = shape.size() - 1; i >= 0; --i) {
              strides[i] = stride;
              stride *= shape[i];
            }
            // Create a numpy array with the given shape and strides.
            py::array_t<double> np_arr(shape, strides);
            auto buf = np_arr.request();
            double *ptr = static_cast<double *>(buf.ptr);
            size_t numElements = cpuStorage->rows() * cpuStorage->cols();
            std::copy(cpuStorage->matrix.data(),
                      cpuStorage->matrix.data() + numElements, ptr);
            return np_arr;
          },
          "Return the tensor as a numpy array with the proper shape.")

      // Bind the operator overloads.
      .def(
          "__add__",
          [](const Tensor<double> &a, const Tensor<double> &b) {
            return a + b;
          },
          "Element-wise addition of two Tensors.")
      .def(
          "__sub__",
          [](const Tensor<double> &a, const Tensor<double> &b) {
            return a - b;
          },
          "Element-wise subtraction of two Tensors.")
      .def(
          "__mul__",
          [](const Tensor<double> &a, const Tensor<double> &b) {
            return a * b;
          },
          "Element-wise multiplication of two Tensors.")
      .def(
          "__truediv__",
          [](const Tensor<double> &a, const Tensor<double> &b) {
            return a / b;
          },
          "Element-wise division of two Tensors.")

      // Bind member functions.
      .def("__matmul__", &Tensor<double>::matmul,
           "Matrix multiplication of two Tensors.")
      // Expose the scalar overload of pow.
      .def("pow",
           py::overload_cast<const double>(&Tensor<double>::pow, py::const_),
           "Raise each element of the Tensor to the given scalar power.")

      // Expose the tensor overload of pow.
      .def("pow",
           py::overload_cast<const Tensor<double> &>(&Tensor<double>::pow,
                                                     py::const_),
           "Raise each element of the Tensor to the elementwise power given by "
           "another Tensor.")
      .def("sqrt", &Tensor<double>::sqrt,
           "Element-wise square root of the Tensor.")
      .def("exp", &Tensor<double>::exp,
           "Element-wise exponential of the Tensor.")
      .def("log", &Tensor<double>::log,
           "Element-wise natural logarithm of the Tensor.")
      .def("sum", &Tensor<double>::sum, "Sum of all elements in the Tensor.")
      .def("transpose", &Tensor<double>::transpose, "Transpose of the Tensor.")
      .def("maximum", &Tensor<double>::maximum,
           "Element-wise maximum between this Tensor and another.");
}
