#include "../matrix.hpp"
#include "../neural_network.hpp"

#include <iostream>
#include <vector>

int main()
{

    // init neural network
    auto nn = numeric::init_neural_network({2, 2, 1});

    // feed forward
    auto ff = numeric::feedforward({1.0f, 1.0f}, nn);
    auto as = std::get<0>(ff);
    auto bs = std::get<1>(ff);

    // back propagation
    auto nn2 = numeric::backpropagation({1.0f, 1.0f}, {0.0f}, nn);
}
