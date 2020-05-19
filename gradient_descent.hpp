#pragma once

#include "derivative.hpp"

#include <algorithm>
#include <math.h>

namespace numeric {

	/*!
	 *
	 */
	std::vector<double> gradient_descent(std::function<double(std::vector<double>)> f, std::vector<double> initial_xs){

		// learning rate
		auto learning_rate = 1.0;

		// vector of xs
		std::vector<double> xs = initial_xs;

		// vector of partial derivatives
		std::vector<double> ds = initial_xs;

		// iterations of gradient descent
		auto best_xs = xs;
		auto best_y = f(xs);
		auto N = 10000;
		for(int j=0;j<N;j++){

			// calculate partial derivatives
			for(int i=0;i<xs.size();i++){ ds[i] = partial_derivative(f)(xs, i); }

			// update
			for(int i=0;i<xs.size();i++){ xs[i] -= learning_rate * ds[i]; }

			// update learning rate if needed
			auto y = f(xs);
			if(y > best_y){
				// slightly smaller learning rate
				learning_rate *= 0.99;
				N++;
			} else {
				// store values
				best_y = y;
				best_xs = xs;
			}

			// sanity
			learning_rate = std::min(std::max(0.0, learning_rate), 1.0);

			// escape
			if(learning_rate < pow(10,-16)){
				break;
			}
		}

		// return
		return best_xs;

	}

	/*!
	 *
	 */
	double gradient_descent(std::function<double(double)> f, double initial_x){
		auto f2 = [f](std::vector<double> xs){ return f(xs[0]); };
		return gradient_descent(f2, {initial_x})[0];
	}

}
