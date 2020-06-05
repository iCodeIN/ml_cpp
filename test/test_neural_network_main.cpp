#include "../gradient_descent.hpp"
#include "../matrix.hpp"
#include "../neural_network.hpp"

#include <iostream>
#include <vector>

int main()
{

    // init neural network
    auto nn = nn::init_neural_network({2, 3, 3, 1});

    // xor inputs
    std::vector<std::vector<float>> xs =
    {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    std::vector<std::vector<float>> ys = {{0.0f},{1.0f},{1.0f},{0.0f}};

    // forward pass
    auto as = std::get<0>(nn::feedforward(xs, nn));
    std::cout << "Before training :" << std::endl;
    matrix::print_matrix(as[as.size()-1]);

    // back propagation
    auto nn2 = nn;
    for(int i=0; i<10; i++)
    {
        nn2 = nn::train(xs, ys, nn2, numeric::constant_learning_rate(0.1f), 1000);
        as = std::get<0>(nn::feedforward(xs, nn2));
        std::cout << "After training :" << std::endl;
        matrix::print_matrix(as[as.size()-1]);
    }
}
