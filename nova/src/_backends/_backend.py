from ._fusion import FusionBackend
from ._numpy import NumpyBackend


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
            getattr(FusionBackend(), 'fusion')
        except ImportError:
            return FusionBackend()
        else:
            return NumpyBackend()
