import numpy as np


def finite_difference(f, x, epsilon):
    """
    fn to calculate numerical derivative using finite difference approximation

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


def f(x):
    return x**2


if __name__ == "__main__":
    x = 10
    print(finite_difference(f, x, 1.5e-8))
