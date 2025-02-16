# threads

## TO DO 
- Add broadcasting
- extend tensor ops (make list of needed ops)
- Remove naive recursion from autodiff in favour of graph engine that performs single topological sort from output to input
- Implement numeric gradient checker to verify correctness backward pass for each operation
- Implement layers module
- Add activation functions
- Add optimiser library
- Add multithreading cpu support
  - c++ module?
- Add GPU support
  - apple metal API?
  - cuda for NVIDIA GPUs