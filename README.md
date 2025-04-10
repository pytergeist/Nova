# Nova: A Deep Learning Framework

Nova is a modular deep learning framework currently under active development, the framework is being designed for ease of use in research.
The high-level API is built in python to simplify building, training, and deploying neural network models. The compute intensive
parts of the framework will be implemented in C++ for performance (the fusion package), which is still under development.

## Features

- **High-Level Python API:**  
  Simplified interface for designing neural networks, managing layers, configuring models, and handling training routines.

- **Modular Architecture:**  
  - **_Backends:** Currently supports multiple computation backends for development purposes - eventually fusion will be the only backend.
  - **Blocks & Models:** Provides pre-built building blocks and sample models.
  - **Automatic Differentiation & Computational Graphs:** Support for creating and managing computation graphs with automatic differentiation capabilities.

- **Documentation:**  
  Documentation will eventually be available for all modules, including tutorials and API references. The python API docs will be generated using Sphinx, and the C++ API docs will be generated using Doxygen.

## Directory Structure

Below is an overview of the nova project’s directory structure, those marked with `# dev` are under development and may not be fully functional yet:
```plaintext
nova
└── src
    ├── _backends
    ├── backend # dev
    │   ├── autodiff
    │   ├── core
    │   │   ├── _C
    │   │   └── dtypes # dev
    │   ├── graph 
    │   ├── operations
    │   └── trainers # dev
    ├── blocks
    │   └── activations # dev
    ├── initialisers 
    └── models # dev
```

## Installation 
Coming soon!

## Contributing
Coming soon!

