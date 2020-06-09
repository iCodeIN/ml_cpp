#pragma once

#include "matrix.hpp"

#include <tuple>
#include <vector>

namespace nn
{

    std::vector<matrix::FloatMatrix> train(
        std::vector<long> sequence,
        std::vector<matrix::FloatMatrix> weights,
        int window_size = 5)
    {
        // window view on sequence
        for(int i=window_size; i<sequence.size() - window_size; i++)
        {
            // store samples
            auto samples = std::vector<std::tuple<long,long,bool>>();
            // positive samples in window
            for(int j=-window_size; j<window_size; j++)
            {

                // check out of bounds
                if(i+j < 0)
                {
                    continue;
                }
                if(i+j >= sequence.size())
                {
                    continue;
                }

                // idempotency
                if(j==0)
                {
                    continue;
                }

                // add sample
                samples.push_back(std::make_tuple(sequence[i], sequence[i+j], true));
            }
            // negative samples

        }

    }

}
