#pragma once

#include <functional>
#include <vector>

namespace numeric {

	/*!
	 *
	 */
	std::function<double(std::vector<double>,int)> partial_derivative(std::function<double(std::vector<double>)> f){
		return [f](std::vector<double> xs, int var_index){
			auto eps = 0.0001;

			std::vector<double> xs_mod_0 = xs;
			xs_mod_0[var_index] += eps;

			std::vector<double> xs_mod_1 = xs;
			xs_mod_1[var_index] -= eps;

			return ( f(xs_mod_0) - f(xs_mod_1) ) / (2 * eps);
		};
	}

	/*!
	 *
	 */
	std::function<double(double)> derivative(std::function<double(double)> f){
		return [f](double x){
			auto eps = 0.0001;
			return ( f(x + eps) - f(x - eps) ) / (2 * eps);
		};
	}
}
