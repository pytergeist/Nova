# import numpy as np
# import pytest
#
# from abditus.tensor import Tensor
# from tests.integration.gradients.finite_difference import finite_difference_jacobian
#
# import logging
#
# logging.basicConfig(
#     filename='app.log',
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     force=True  # This forces the configuration, overriding existing handlers.
# )
#
# # Write an informational log message
# logging.info('This is an informational message')
#
#
# # TODO: Add parameterisation for multiple test cases
#
#
# def fn_numpy(x):
#     return x @ x
#
#
# def compute_autodiff_gradient(x):
#     x_tensor = Tensor(x, requires_grad=True)
#
#     output = x_tensor @ x_tensor
#
#     output.backward()
#
#     return x_tensor.grad
#
#
# def test_matmul_grad():  # TODO: Should there be a test for matmul or 1D arrays?
#     x_test = np.random.rand(10, 10)
#
#     numerical_jacobian = finite_difference_jacobian(fn_numpy, x_test, epsilon=1.5e-8)
#     logging.info(f"numerical_jacobian shape: {numerical_jacobian.shape}")
#     numerical_jacobian = numerical_jacobian.squeeze()
#     logging.info(f"numerical_jacobian post squeeze shape: {numerical_jacobian.shape}")
#     numerical_vector_grad = np.dot(np.ones(x_test.shape), numerical_jacobian)
#     analytical_grad = compute_autodiff_gradient(x_test)
#
#     np.testing.assert_allclose(
#         analytical_grad,
#         numerical_vector_grad,
#         rtol=1e-5,
#         atol=1e-7,
#         err_msg="Autodiff gradient does not match numerical gradient for addition.",
#     )
#
#
# if __name__ == "__main__":
#     pytest.main([__file__])
