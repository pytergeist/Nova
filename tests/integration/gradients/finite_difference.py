import numpy as np


def finite_difference(f, x, epsilon):
    """
    fn to calculate numerical derivative using finite difference approximation
    This fn only computes scalar gradients for functions, not vectors

    f'(x_i) = (f(x + ε*e_i) - f(x - ε*e_i)) / 2*ε)
    """
    if np.isscalar(x) or np.ndim(x) == 0:
        return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

    x = np.asarray(x)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = np.zeros_like(x)
        dx[i] = epsilon
        grad[i] = (f(x + dx) - f(x - dx)) / (2 * epsilon)
    return grad


def finite_difference_jacobian(f, x, epsilon=1.5e-8):
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    f0 = np.atleast_1d(np.asarray(f(x), dtype=np.float64)).ravel()
    jacobian = np.zeros((f0.size, x.size), dtype=np.float64)

    for i in range(x.size):
        dx = np.zeros_like(x, dtype=np.float32)
        dx[i] = epsilon

        f_pos = np.atleast_1d(np.array(f(x + dx), dtype=np.float64)).ravel()
        f_neg = np.atleast_1d(np.array(f(x - dx), dtype=np.float64)).ravel()

        jacobian[:, i] = (f_pos - f_neg) / (2*epsilon)

    return jacobian


def f(x):
    return [x**2, x**11]


if __name__ == "__main__":
    x = [10, 10, 10]
    print(finite_difference_jacobian(f, x, 1.5e-8))
