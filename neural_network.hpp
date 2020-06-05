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
            // matrix multiplication
            bs.push_back(matrix::mul(as[as.size() - 1], weights[i]));

            // logistic function
            as.push_back(matrix::apply_function(bs[bs.size()-1], activation_function));
        }

        // output
        return std::make_tuple(as, bs);
    }

    std::vector<matrix::FloatMatrix> backpropagation(
        const std::vector<float>& xs,
        const std::vector<float>& ys,
        const std::vector<matrix::FloatMatrix>& weights,
        float learning_rate = 0.1)
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
                delta = matrix::dotproduct(matrix::mul(deltas[0], matrix::transpose(weights[i])), as_derivatives[i]);
            }
            deltas.insert(deltas.begin(), delta);
        }

        // update weight(s)
        std::cout << std::endl;
        auto weights_out = std::vector<matrix::FloatMatrix>();
        for(int i = 1 ; i < deltas.size() ; i++ )
        {
            auto weight_update = matrix::scalar(matrix::mul(matrix::transpose(as[i-1]), deltas[i]), learning_rate);
            weights_out.push_back(matrix::add(weights[i - 1], weight_update));
        }

        // return
        return weights_out;

    }

    std::vector<matrix::FloatMatrix> train(
        const std::vector<float> xs,
        const std::vector<float> ys,
        const std::vector<matrix::FloatMatrix>& initial_weights)
    {
        auto w = backpropagation(xs, ys, initial_weights);
        for(int i=0; i<1000; i++)
        {
            w = backpropagation(xs, ys, w, 0.1f);
            auto ys_out = std::get<0>(feedforward(xs, w));
            matrix::print_matrix(ys_out[ys_out.size()-1]);
        }
        return w;
    }

}
