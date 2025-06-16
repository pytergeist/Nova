from typing import Optional, Tuple


class InputBlock:
    def __init__(
        self,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[str] = None,
    ):
        self.shape = shape
        self._dtype = dtype
        # TODO: Add validation checks

    @property
    def dtype(self) -> Optional[str]:
        """Return the data type of the input block."""
        return self._dtype

    def call(self, *args, **kwargs):
        return

    def get_config(self) -> dict:
        """Return the configuration of the input block."""
        return {
            "shape": self.shape,
            "dtype": self.dtype,
        }
