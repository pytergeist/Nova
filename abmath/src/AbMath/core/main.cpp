#include "tensor.h"


int main() {
    Tensor const tensor1({1.0, 2.0, 3.0, 4.0, 5.0});
    Tensor const tensor2({1.0, 2.0, 3.0, 4.0, 5.0});
    Tensor const &tensor3 = tensor1 / tensor2;
    std::cout << tensor3 << std::endl;
    return 0;
}
