import numpy as np


def finite_difference_jacobian(f, x, epsilon=1.5e-8):
    """Compute an approximate Jacobian matrix using the finite difference method.

    This function now supports inputs where x can be a multi-dimensional array.
    The Jacobian is computed with respect to the flattened version of x.

    Args:
        f (callable): A function that accepts an array x and returns an array.
        x (scalar or array-like): Input value(s).
        epsilon (float, optional): Perturbation size. Defaults to 1.5e-8.

    Returns:
        numpy.ndarray: The approximate Jacobian with shape (f(x).size, x.size)
    """
    x = np.asarray(x, dtype=np.float32)
    f0 = np.atleast_1d(np.asarray(f(x), dtype=np.float32)).ravel()
    jacobian = np.zeros((f0.size, x.size), dtype=np.float32)

    for flat_idx, multi_idx in enumerate(np.ndindex(x.shape)):
        dx = np.zeros_like(x, dtype=np.float32)
        dx[multi_idx] = epsilon

        f_pos = np.atleast_1d(np.asarray(f(x + dx), dtype=np.float32)).ravel()
        f_neg = np.atleast_1d(np.asarray(f(x - dx), dtype=np.float32)).ravel()

        jacobian[:, flat_idx] = (f_pos - f_neg) / (2 * epsilon)

    return jacobian


if __name__ == "__main__":

    def f(x):
        return [x, x]

    x = [10, 10, 10]

    numerical_jacobian = finite_difference_jacobian(f, x, epsilon=1.5e-8)
    print(numerical_jacobian)
    print(numerical_jacobian.shape)
