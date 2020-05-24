#pragma once

#include <assert.h>
#include <functional>
#include <vector>

#include "linear_regression.hpp"

namespace numeric {

	/*!
	 * Polynomial Regression is a form of linear regression in which 
	 * the relationship between the independent variable x and dependent variable y 
	 * is modeled as an nth degree polynomial.
	 */
	std::vector<float> polynomial_regression(
		const std::vector<float>& xs, 	//! xs datapoints
		const std::vector<float>& ys, 	//! ys datapoints
		int degree_of_polynomial	//! maximum degree of the polynomial to fit
		){

		assert(degree_of_polynomial >= 0);
		assert(xs.size() > 0);
		assert(ys.size() > 0);;
		assert(xs.size() == ys.size());


		// build loss function
		auto mae_loss_function = [](std::vector<float> pred_ys, std::vector<float> ys){
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
		auto poly_pred_function = [](std::vector<float> coeffs, std::vector<float> xs){
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
		std::vector<float> coeffs;
		for(int i=0;i<=degree_of_polynomial;i++){coeffs.push_back(0.0);}

		// pack xs into mtx
		std::vector<std::vector<float>> mtx_xs;
		for(int i=0;i<xs.size();i++){mtx_xs.push_back({xs[i]});}

		// delegate
		return linear_regression(mtx_xs,
					ys,
					poly_pred_function,
					coeffs,
					mae_loss_function);
	}

}
