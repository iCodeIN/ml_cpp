#pragma once

#include "derivative.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <math.h>

namespace numeric {

	std::function<float(int)> step_decay_learning_rate(float initial_learning_rate = 1.0f, float decay = 0.5f, int epochs_drop = 10){
		return [decay, initial_learning_rate, epochs_drop](int iteration_nr){

			// setup
			if(iteration_nr == 0){return initial_learning_rate;}

			// step decay learning rate formula
			auto a = (1.0f + iteration_nr) / epochs_drop;
			
			// return
			return initial_learning_rate * pow(decay, a);
		};
	}

	/*!
	 * Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. 
	 * To find a local minimum of a function using gradient descent, we take steps proportional to the negative of the gradient 
	 * (or approximate gradient) of the function at the current point. 
	 */
	std::vector<float> gradient_descent(
			const std::function<float(std::vector<float>)>& f, 		//! function to perform gradient descent on
			const std::vector<float>& initial_xs, 				//! initial guess for the (local) minimum
			const std::function<float(int)>& learning_rate_schedule,	//! function that determines the learning rate based on the iteration nr
			int max_number_of_iterations = 16348,				//! maximum number of iterations
			bool stop_when_partial_derivative_is_zero = true,		//! flag to determine whether to stop when the partial derivative is zero
			bool stop_when_learning_rate_is_too_small = true		//! flag to determine whether to stop when the learning rate becomes too small
			){

		// learning rate
		auto learning_rate = learning_rate_schedule(0);

		// vector of xs
		std::vector<float> xs = initial_xs;

		// vector of partial derivatives
		std::vector<float> ds = initial_xs;

		// iterations of gradient descent
		auto best_xs = xs;
		auto best_y = f(xs);
		for(int j=0;j<max_number_of_iterations;j++){

			// calculate partial derivatives
			for(int i=0;i<xs.size();i++){
				ds[i] = partial_derivative(f)(xs, i); 
			}

			// update
			for(int i=0;i<xs.size();i++){ xs[i] -= learning_rate * ds[i]; }

			// update learning rate if needed
			auto y = f(xs);
			if(y < best_y){
				best_y = y;
				best_xs = xs;
			}

			// ensure learning rate is reasonably limited [0 .. 1]
			learning_rate = learning_rate_schedule(j);
			assert(learning_rate >= 0);

			// escape optimization loop if learning rate is too small
			if(stop_when_learning_rate_is_too_small && learning_rate < pow(10,-16)){
				break;
			}

			// escape optimization loop if all partial derivatives are zero
			auto all_ds_zero = true;
			for(int k=0;k<ds.size();k++){ all_ds_zero &= (ds[k] == 0);}
			if(stop_when_partial_derivative_is_zero && all_ds_zero){
				break;
			}

		}

		// return
		return best_xs;

	}

	/*!
	 * Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. 
	 * To find a local minimum of a function using gradient descent, we take steps proportional to the negative of the gradient 
	 * (or approximate gradient) of the function at the current point. 
	 */
	float gradient_descent(const std::function<float(float)>& f, float initial_x){

		// turn single argument into vector
		auto multi_arg_f = [&f](std::vector<float> xs){ return f(xs[0]); };
	
		// delegate
		return gradient_descent(multi_arg_f, 
					{initial_x}, 
					step_decay_learning_rate(1.0f, 0.5f, 1024),
					16384)[0];
	}

}
