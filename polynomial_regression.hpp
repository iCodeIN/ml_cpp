#pragma once

#include "linear_regression.hpp"

namespace numeric {

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
