
#pragma once

#include <assert.h>
#include <functional>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

#include "gradient_descent.hpp"

namespace numeric
{

    /*!
     * In statistics, linear regression is a linear approach to modeling the relationship
     * between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables).
     * The case of one explanatory variable is called simple linear regression.
     */
    std::vector<float> linear_regression(
        const std::vector<std::vector<float>>& xs,								//! xs datapoints
        const std::vector<float>& ys,										//! ys datapoints
        const std::function<float(std::vector<float>, std::vector<float>)>& pred_function,			//! function that attempts to predict relationship between xs and ys
        const std::vector<float>& initial_params,								//! initial parameters for the prediction function
        const std::function<float(std::vector<float>, std::vector<float>)>& loss_function,			//! loss function (cost the algorithm has to pay for bad predictions)
        const std::function<float(int)>& learning_rate_schedule = step_decay_learning_rate(1.0f, 0.5f, 1024),	//! learning rate schedule (passed to gradient descent)
        int max_number_of_iterations = 16384,									//! maximum number of iterations (passed to gradient descent)
        int batch_size = -1											//! batch size
    )
    {

        // asserts
        assert(xs.size() > 0);
        assert(ys.size() > 0);
        assert(xs.size() == ys.size());
        assert(batch_size == -1 || batch_size > 0);
        for(int i=0; i<xs.size(); i++)
        {
            assert(xs[0].size() == xs[i].size());
        }

        // obtain iteration nr
        auto iteration_nr = 0;
        auto lrs = [&learning_rate_schedule, &iteration_nr](int i)
        {
            iteration_nr = i;
            return learning_rate_schedule(i);
        };

        // build function to be passed to gradient descent
        if(batch_size == -1)
        {
            batch_size=xs.size();
        }
        auto f = [&pred_function, &loss_function, &xs, &ys, &iteration_nr, batch_size](std::vector<float> params)
        {

            // batch logic here
            auto start_index = (iteration_nr * batch_size) % ys.size();
            auto stop_index = std::min(start_index + batch_size, ys.size());

            // run prediction function
            std::vector<float> ys_t;		// truth
            std::vector<float> ys_h;		// hypothesis
            for(int i=start_index; i!=stop_index; i=( i + 1 % ys.size()) )
            {
                auto y_t = ys[i];
                auto y_h = pred_function(params, xs[i]);
                ys_t.push_back(y_t);
                ys_h.push_back(y_h);
            }

            // run loss function
            auto loss = loss_function(ys_t, ys_h);

            // return
            return loss;
        };

        // pass function to gradient descent
        auto out_params = gradient_descent(f, initial_params, lrs, max_number_of_iterations);

        /*
         * try a small range of 'pretty' coefficients near the ones that gradient descent found
         */
        auto N = 1;
        std::vector<std::vector<float>> pretty_params_ops;
        for(int i=0; i<out_params.size(); i++)
        {
            auto p = out_params[i];
            if(p == floor(p))
            {
                pretty_params_ops.push_back({p});
            }
            else
            {
                pretty_params_ops.push_back(
                {
                    floor(p),	// p rounded up
                    ceil(p), 	// p rounded down
                    p
                });		// p itself
                N *= 3;
            }
        }
        std::vector<int> M;
        for(int i=0; i<out_params.size(); i++)
        {
            M.push_back(0);
        }
        for(int i=0; i<N; i++)
        {

            // build 'pretty' params
            auto pretty_params = out_params;
            for(int j=0; j<M.size(); j++)
            {
                pretty_params[j] = pretty_params_ops[j][M[j]];
            }

            // calculate loss
            auto loss = f(pretty_params);
            if(loss < f(out_params))
            {
                out_params = pretty_params;
            }

            // update index (moving to next configuration of pretty params
            M[0]++;
            for(int j=0; j<M.size(); j++)
            {
                if(M[j] == pretty_params_ops[j].size() && j < M.size() - 1)
                {
                    M[j] = 0;
                    M[j+1]++;
                }
            }
        }

        // return
        return out_params;
    }

}
