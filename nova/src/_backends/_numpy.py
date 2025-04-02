import numpy as np


class NumpyBackend:
    def __init__(self):
        pass

    @staticmethod
    def add(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return np.add(v1, v2)

    @staticmethod
    def subtract(v1, v2) -> np.ndarray:
        return np.subtract(v1, v2)

    @staticmethod
    def multiply(v1, v2) -> np.ndarray:
        return np.multiply(v1, v2)

    @staticmethod
    def divide(v1, v2) -> np.ndarray:
        return np.divide(v1, v2)

    @staticmethod
    def matmul(v1, v2) -> np.ndarray:
        return np.matmul(v1, v2)

    @staticmethod
    def sum(v1: np.ndarray) -> np.ndarray:
        return np.sum(v1)

    @staticmethod
    def maximum(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return np.maximum(v1, v2)

    @staticmethod
    def transpose(v1: np.ndarray) -> np.ndarray:
        return np.transpose(v1)

    @staticmethod
    def power(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return np.power(v1, v2)

    @staticmethod
    def exp(v1: np.ndarray) -> np.ndarray:
        return np.exp(v1)

    @staticmethod
    def log(v1: np.ndarray) -> np.ndarray:
        return np.log(v1)

    @staticmethod
    def sqrt(v1: np.ndarray) -> np.ndarray:
        return np.sqrt(v1)
