class Loss:
    """
    Base class for loss functions.
    """

    def __call__(self, *args, **kwargs):
        """
        Call the loss function with the provided arguments.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def __repr__(self):
        """
        Return a string representation of the loss function.
        """
        return f"LossFunction :{self.__class__.__name__}()"
