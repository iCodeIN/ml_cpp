#pragma once

#include <assert.h>
#include <functional>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include "linear_regression.hpp"

namespace numeric
{

    /*!
     * In statistics, the logistic model (or logit model) is used to model the
     * probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick.
     * This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc.
     * Each object being detected in the image would be assigned a probability between 0 and 1 and the sum adding to one.
     */
    std::vector<float> logistic_regression(
        const std::vector<std::vector<float>>& xs,	//! xs datapoints
        const std::vector<float>& ys			//! ys datapoints (zero or one)
    )
    {

        // asserts
        assert(xs.size() > 0);
        assert(ys.size() > 0);
        assert(xs.size() == ys.size());
        for(int i=0; i<xs.size(); i++)
        {
            assert(xs[0].size() == xs[i].size());
        }

        // prediction function
        auto pred_function = [](std::vector<float> coeffs, std::vector<float> xs)
        {
            auto h = 0.0;
            for(int i=0; i<xs.size(); i++)
            {
                h += xs[i] * coeffs[i];
            }
            h += coeffs[coeffs.size() - 1];
            auto z = 1.0f / (1.0f + exp(-h));
            return z;
        };

        // loss function
        auto loss_function = [&pred_function](std::vector<float> ys, std::vector<float> pred_ys)
        {
            auto loss = 0.0;
            for(int i=0; i<ys.size(); i++)
            {
                auto pred_y = std::min(std::max(pred_ys[i], 0.001f), 0.999f);
                loss += (-ys[i] * log(pred_y) - (1.0f - ys[i]) * log(1.0f - pred_y));
            }
            loss /= ys.size();
            return loss;
        };

        // build initial params
        srand(time(NULL));
        std::vector<float> coeffs;
        for(int i=0; i<=xs[0].size(); i++)
        {
            auto p = (rand() % 100) / 1000.0f;
            coeffs.push_back(0.0f);
        }

        // delegate
        return linear_regression(xs, ys, pred_function, coeffs, loss_function, step_decay_learning_rate(0.9f, 0.99f, 128), 16348);

    }
}
