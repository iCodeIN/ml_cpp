
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
		auto out_params = gradient_descent(f, initial_params);

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

	/*!
	 * Polynomial Regression is a form of linear regression in which 
	 * the relationship between the independent variable x and dependent variable y 
	 * is modeled as an nth degree polynomial.
	 */
	std::vector<double> polynomial_regression(std::vector<double> xs, std::vector<double> ys, int degree_of_polynomial){

		assert(degree_of_polynomial >= 0);

		// build loss function
		auto mae_loss_function = [](std::vector<double> pred_ys, std::vector<double> ys){
			assert(pred_ys.size() == ys.size());
			auto k = 0.0;
			for(int i=0;i<pred_ys.size();i++){
				auto j = abs(pred_ys[i] - ys[i]);
				k += j;
			}
			k /= pred_ys.size();
			return k;
		};

		// build pred function
		auto poly_pred_function = [](std::vector<double> coeffs, std::vector<double> xs){
			assert(xs.size() == 1);
			assert(coeffs.size() >= 1);
			auto x = xs[0];
			auto y = 0.0;
			for(int i=0;i<coeffs.size();i++){
				y += pow(x, i) * coeffs[i];
			}
			return y;
		};

		// build initial params
		std::vector<double> coeffs;
		for(int i=0;i<=degree_of_polynomial;i++){coeffs.push_back(0.0);}

		// pack xs into mtx
		std::vector<std::vector<double>> mtx_xs;
		for(int i=0;i<xs.size();i++){mtx_xs.push_back({xs[i]});}

		// delegate
		return linear_regression(mtx_xs,
					ys,
					poly_pred_function,
					coeffs,
					mae_loss_function);
	}

}
