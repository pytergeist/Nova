import sys

from nova.src._backends._fusion import FusionBackend
from nova.src._backends._numpy import NumpyBackend

sys.path.append(
    "/Users/tompope/Documents/Documents - Tom’s MacBook Air/toms_personal_devs/deep_learning/Nova/fusion/build"
)


class Backend:
    """Backend class that delegates to the appropriate backend.
    This class is necessary during the development of the library to allow
    the codebase/user to switch between the two backends. The np/fs backends
    will be abstractions on top of the numpy/fusion package - they will serve to simply define the implemented
    elementwise/matrix functions needed throughout development.
    """

    def __init__(self):
        self._backend = None

    def __getattr__(self, name):
        if self._backend is None:
            self._backend = self._get_backend()
        return getattr(self._backend, name)

    @staticmethod
    def _get_backend():
        try:
            fusion_backend = FusionBackend()
            assert getattr(fusion_backend, "backend") is not None
            return fusion_backend
        except AssertionError:
            numpy_backend = NumpyBackend()
            assert getattr(numpy_backend, "backend") is not None
            return numpy_backend


if __name__ == "__main__":
    backend = Backend()
    bb = backend._get_backend()
    print(bb)
