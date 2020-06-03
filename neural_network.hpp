#pragma once

#include <assert.h>
#include <functional>
#include <iostream>
#include <tuple>
#include <vector>

#include "matrix.hpp"

namespace numeric
{

    std::vector<matrix::FloatMatrix> init_neural_network(std::vector<int> layer_sizes)
    {
        assert(layer_sizes.size() >= 2);
        std::vector<matrix::FloatMatrix> mtx;
        for(int i=0; i<layer_sizes.size() - 1; i++)
        {
            auto m = layer_sizes[i];
            auto n = layer_sizes[i+1];
            mtx.push_back(matrix::random(m, n));
        }
        return mtx;
    }

    std::tuple<std::vector<matrix::FloatMatrix>, std::vector<matrix::FloatMatrix>> feedforward(const std::vector<float>& xs, const std::vector<matrix::FloatMatrix>& weights)
    {

        std::vector<matrix::FloatMatrix> as;
        std::vector<matrix::FloatMatrix> bs;

        // define activation function
        auto activation_function = [](float x)
        {
            return 1.0f / (1.0f + exp(-x));
        };

        // convert input to matrix
        auto xs_mtx = matrix::FloatMatrix();
        xs_mtx.push_back(xs);
        as.push_back(xs_mtx);

        // run the input through all layers
        for(int i = 0 ; i < weights.size() ; i++)
        {

            // debug
            std::cout << "as[" << as.size() - 1 << "]" << std::endl;
            matrix::print_matrix(as[as.size() - 1]);
            std::cout << std::endl;

            std::cout << "weights[" << i << "]" << std::endl;
            matrix::print_matrix(weights[i]);
            std::cout << std::endl;

            // matrix multiplication
            bs.push_back(matrix::mul(as[as.size() - 1], weights[i]));

            std::cout << "bs[" << bs.size() - 1 << "]" << std::endl;
            matrix::print_matrix(bs[bs.size() - 1]);
            std::cout << std::endl;

            // logistic function
            as.push_back(matrix::apply_function(bs[bs.size()-1], activation_function));
        }

        // debug
        std::cout << "as[" << as.size() - 1 << "]" << std::endl;
        matrix::print_matrix(as[as.size() - 1]);
        std::cout << std::endl;


        // output
        return std::make_tuple(as, bs);
    }

    std::vector<matrix::FloatMatrix> backpropagation(const std::vector<float>& xs, const std::vector<float>& ys, const std::vector<matrix::FloatMatrix>& weights)
    {

        // pow2 function
        auto pow2div2 = [](float x)
        {
            return x * x / 2.0f;
        };

        // derivative of transfer function
        auto activation_derivative_function = [](float x)
        {
            return x * (1.0f - x);
        };

        // convert ys to mtx
        auto ys_mtx = {ys};

        // activations and transfers
        auto tpl = feedforward(xs, weights);
        auto as = std::get<0>(tpl);
        auto bs = std::get<1>(tpl);

        // activation function derivative(s)
        auto activation_function_derivative = [](float x)
        {
            return x * (1.0f - x);
        };
        auto as_derivatives = std::vector<matrix::FloatMatrix>();
        for(int i=0; i<as.size(); i++)
        {
            as_derivatives.push_back(matrix::apply_function(as[i], activation_function_derivative));
        }

        // delta(s)
        auto deltas = std::vector<matrix::FloatMatrix>();
        for(int i=as.size() - 1 ; i >= 0 ; i--)
        {
            auto delta = matrix::FloatMatrix();
            if(i == as.size() - 1)
            {
                delta = matrix::dotproduct(matrix::subtract(ys_mtx, as[i]), as_derivatives[i]);
            }
            else
            {
                delta = matrix::dotproduct(matrix::mul(deltas[deltas.size() - 1], matrix::transpose(weights[i])), as_derivatives[i]);
            }

            // debug
            std::cout << "delta[" << i << "]" << std::endl;
            matrix::print_matrix(delta);

            // store delta
            deltas.push_back(delta);
        }

        // update weight(s)
        auto updates = std::vector<matrix::FloatMatrix>();
        for(int i=0; i<weights.size(); i++)
        {
            // TODO : verify indices
            auto weight_update = matrix::scalar(matrix::mul(matrix::transpose(as[i]), deltas[i]), learning_rate);
            weight_updates.push_back(weight_update);
        }

        // return
        return weights;

    }

}
