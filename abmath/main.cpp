
#include "src/AbMath/templates/operations.h"
#include "src/AbMath/templates/vector.h"
#include <iostream>

int main() {
  // Create two vectors of doubles with 5 elements each.
  abmath::Vector<double> v1(5), v2(5);

  // Initialize vectors.
  for (std::size_t i = 0; i < 5; ++i) {
    v1[i] = static_cast<double>(i);
    v2[i] = static_cast<double>(i * 2);
  }

  // Add two vectors.
  auto v3 = abmath::add(v1, v2);

  // Add a vector and a scalar.
  auto v4 = abmath::add(v1, 3.0);

  // Add a scalar and a vector.
  auto v5 = abmath::add(3.0, v2);

  // Add two scalars.
  double a = 2.0, b = 4.0;
  double c = abmath::add(a, b);

  // Print the results.
  std::cout << "v1 + v2 = ";
  v3.print();

  std::cout << "v1 + 3.0 = ";
  v4.print();

  std::cout << "3.0 + v2 = ";
  v5.print();

  std::cout << "2.0 + 4.0 = " << c << std::endl;

  return 0;
}
