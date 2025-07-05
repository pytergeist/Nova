from typing import Optional

from nova.src.backend.topology import Builder


class Trainer:
    def __init__(self, builder: Optional[Builder] = None):
        self.builder = builder or Builder.get_current()
        self._topology = None
        self._compiled = False

    @property
    def topology(self):
        if self._topology:
            return self._topology
        raise ValueError(
            "Model topology has not yet been built, please call the .build() method"
        )

    @topology.setter
    def topology(self, value):
        """Set the model graph for the trainer."""
        if not isinstance(value, list):
            raise ValueError("Model graph must be a list of nodes.")
        self._topology = value

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

        self.topology = self.builder.sort_model_graph()

        for node in self.topology:
            if not node.operator.built:
                node.operator.build_block(node.operator.input_shape)
            for child in node.children:
                child.operator.input_shape = node.operator.output_shape

    def compile(self):
        """Compile the trainer."""
        if self._compiled:
            raise RuntimeError("Model is already compiled.")

        self._compiled = True
        print("Model compiled successfully.")
