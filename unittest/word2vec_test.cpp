
#include "../neural_network.hpp"
#include "../word2vec.hpp"

int main()
{
    // sequence
    std::vector<long> words = {1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 4, 7, 5, 6, 8, 7, 4, 5};

    // empty neural network
    auto nn = nn::init_neural_network({1000, 32, 1000});

    // word2vec
    word2vec::train(words, nn, 2);

}
