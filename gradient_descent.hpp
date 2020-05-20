#pragma once

#include "derivative.hpp"

#include <algorithm>
#include <math.h>

namespace numeric {

	/*!
	 * Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. 
	 * To find a local minimum of a function using gradient descent, we take steps proportional to the negative of the gradient 
	 * (or approximate gradient) of the function at the current point. 
	 * But if we instead take steps proportional to the positive of the gradient, we approach a local maximum of that function; 
	 * the procedure is then known as gradient ascent. 
	 * Gradient descent was originally proposed by Cauchy in 1847.
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

			// ensure learning rate is reasonably limited [0 .. 1]
			learning_rate = std::min(std::max(0.0, learning_rate), 1.0);

			// escape optimization loop if learning rate is too small
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
