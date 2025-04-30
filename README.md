# Nova: A Deep Learning Framework

Nova is a modular deep learning framework currently under active development, the framework is being designed for ease of use in research, with a pytorch like C++/python integration and high level keras like APIs.
The high-level API is built in python to simplify building, training, and deploying neural network models. The compute intensive
parts of the framework will be implemented in C++ for performance (the fusion package), which is still under development.

## Features

- **High-Level Python API:**  
  Simplified interface for designing neural networks, managing layers, configuring models, and handling training routines.

- **Modular Architecture:**  
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
    │   │   ├── _C # dev
    │   │   └── dtypes # dev
    │   ├── graph 
    │   ├── operations
    │   └── trainers # dev
    ├── blocks
    │   └── activations # dev
    ├── initialisers 
    └── models # dev
```

## Road Map
### Fusion
Currently, I am working on Fusion - right now I am working to provide support for arbitrary size ND tensors, and a set of basic operations (addition, multiplication, etc.) for these tensors. 
The next steps on the Fusion roadmap are:
- Support for basic operations on ND tensors (addition, multiplication, matmul etc.)
- Optimized CBlas routines for matrix operations
- SIMD Routines for elementwise operations
- Once these are complete I will be moving back to working on the python interface (Nova)
- Once a prototype of the python interface is complete I will work on migrating the current autodiff code to Fusion

### Nova
The next steps on the Nova roadmap are:
- Training module for training basic network training
- Model module as a high level abstraction for building networks quickly
- Optimisers module

## Installation 
Coming soon!

## Contributing
Coming soon!

