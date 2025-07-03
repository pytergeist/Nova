class Trainer:
    def __init__(self):
        self._compiled = False

    @property
    def compiled(self):
        """Check if the trainer is compiled."""
        return self._compiled

    @compiled.setter
    def compiled(self, value: bool):
        """Set the compiled status of the trainer."""
        if not isinstance(value, bool):
            raise ValueError("Compiled status must be a boolean value.")
        self._compiled = value

    def build(self):
        from nova.src.blocks.block import (  # TODO: placeholder until builder has been switched to context manager
            builder,
        )

        graph = builder.sort_model_graph()
        self.graph = graph

        for node in graph:
            if not node.operator.built:
                input_shape = node.operator.input_shape
                if not hasattr(node.operator, "input_block"):
                    node.operator._build(
                        input_shape
                    )  # TODO: change method to non-protected
                else:
                    node.operator.output_shape = node.operator.input_shape
                    input_shape = node.operator.output_shape

            for child in node.children:
                child.operator.input_shape = node.operator.output_shape

    def compile(self):
        """Compile the trainer."""
        if self._compiled:
            raise RuntimeError("Model is already compiled.")

        self._compiled = True
        print("Model compiled successfully.")
