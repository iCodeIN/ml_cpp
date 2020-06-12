#pragma once

#include "matrix.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <stdlib.h>
#include <time.h>
#include <tuple>
#include <vector>

namespace word2vec
{

    std::vector<matrix::FloatMatrix> train(
        std::vector<long> sequence,
        std::vector<matrix::FloatMatrix> weights,
        int window_size = 5,
        int number_of_negative_samples = 5)
    {

        // frequency tuples
        std::map<long,long> word_frequency;
        for(int i=0; i<sequence.size(); i++)
        {
            ++word_frequency[sequence[i]];
        }
        std::vector<std::pair<long,long>> word_frequency_tuples;
        for(std::pair<long,long> e : word_frequency)
        {
            std::cout << e.first << " : " << e.second << std::endl;
            word_frequency_tuples.push_back(e);
        }

        // total frequency
        auto total_frequency = 0.0f;
        for(int i=0; i<word_frequency_tuples.size(); i++)
        {
            total_frequency += word_frequency_tuples[i].second;
        }
        std::cout << "sum : " << total_frequency << std::endl;

        // sort frequency table
        auto sort_by_val = [](const std::pair<long,long>& t0, const std::pair<long,long>& t1)
        {
            return t0.second < t1.second;
        };
        std::sort(word_frequency_tuples.begin(), word_frequency_tuples.end(), sort_by_val);

        // make 'select random word' function
        std::vector<long> word_lookup_table;
        auto k = word_frequency_tuples.size() - 1;
        auto l = 100000;
        while(word_lookup_table.size() < l)
        {
            auto n = (word_frequency_tuples[k].second / total_frequency) * l;
            for(int i=0; i<n; i++)
            {
                word_lookup_table.push_back(word_frequency_tuples[k].first);
            }
            k--;
        }
        srand(time(NULL));
        auto pick_random_word = [&word_lookup_table]()
        {
            return word_lookup_table[rand() % word_lookup_table.size()];
        };

        // window view on sequence
        for(int i=0; i<sequence.size() - window_size; i++)
        {

            // store samples
            auto samples = std::vector<std::tuple<long, long, float>>();

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
                samples.push_back(std::make_tuple(sequence[i], sequence[i+j], 1.0f));
            }

            // negative samples
            for(int j=0; j<number_of_negative_samples; j++)
            {
                samples.push_back(std::make_tuple(sequence[i], pick_random_word(), 0.0f));
            }

            /*
             * training
             */
            for(int j=0; j<samples.size(); j++)
            {
                std::cout << std::get<0>(samples[j]) << "\t" << std::get<1>(samples[j]) << "\t" << std::get<2>(samples[j]) << std::endl;
            }
        }

        // return
        return weights;
    }

}
