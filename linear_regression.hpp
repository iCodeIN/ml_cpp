
#pragma once

#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "gradient_descent.hpp"

namespace numeric {

	/*!
	 * In statistics, linear regression is a linear approach to modeling the relationship 
	 * between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). 
	 * The case of one explanatory variable is called simple linear regression.
	 */
	std::vector<double> linear_regression(
		std::vector<std::vector<double>> xs,
		std::vector<double> ys,
		std::function<double(std::vector<double>, std::vector<double>)> pred_function,
		std::vector<double> initial_params,
		std::function<double(std::vector<double>, std::vector<double>)> loss_function){

		// asserts
		assert(xs.size() > 0);
		assert(ys.size() > 0);
		assert(xs.size() == ys.size());
		for(int i=0;i<xs.size();i++){assert(xs[0].size() == xs[i].size());}

		// build function to be passed to gradient descent
		auto f = [pred_function, loss_function, xs, ys](std::vector<double> params){

			// run prediction function
			std::vector<double> pred_ys = ys;
			for(int i=0;i<xs.size();i++){
				pred_ys[i] = pred_function(params, xs[i]);
			}

			// run loss function
			auto loss = loss_function(ys, pred_ys);

			// return
			return loss;
		};

		// pass function to gradient descent
		auto out_params = gradient_descent(f, initial_params, 16384);

		// try a small range of 'pretty' coefficients near the ones that gradient descent found
		auto N = 1;
		std::vector<std::vector<double>> pretty_params_ops;
		for(int i=0;i<out_params.size();i++){
			auto p = out_params[i];
			if(p == floor(p)){
				pretty_params_ops.push_back({p});
			} else {
				pretty_params_ops.push_back({
						floor(p),	// p rounded up
						ceil(p), 	// p rounded down
						p});		// p itself
				N *= 3;
			}
		}
		std::vector<int> M;
		for(int i=0;i<out_params.size();i++){M.push_back(0);}
		for(int i=0;i<N;i++){

			// build 'pretty' params
			auto pretty_params = out_params;
			for(int j=0;j<M.size();j++){ pretty_params[j] = pretty_params_ops[j][M[j]]; }

			// calculate loss
			auto loss = f(pretty_params);
			if(loss < f(out_params)){
				out_params = pretty_params;
			}

			// update index (moving to next configuration of pretty params
			M[0]++;
			for(int j=0;j<M.size();j++){
				if(M[j] == pretty_params_ops[j].size() && j < M.size() - 1){
					M[j] = 0;
					M[j+1]++;
				}
			}
		}

		// return
		return out_params;
	}

}
