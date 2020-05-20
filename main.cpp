
#include "derivative.hpp"
#include "gradient_descent.hpp"
#include "linear_regression.hpp"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

void test_derivative(){

	auto sin_f= [](double x){return sin(x);};
	auto cos_f = numeric::derivative(sin_f);

	auto avg_err = 0.0;
	auto max_err = 0;
	auto min_err = 1;
	for(int i=0;i<360;i++){
		auto j = i * 0.01745329;
		auto k = abs(cos_f(j) - cos(j));
		avg_err += k;
		min_err = k < min_err ? k : min_err;
		max_err = k > max_err ? k : max_err;
	}
	avg_err /= 360;

	std::cout << "test_derivative" << std::endl;
	std::cout << "Avg. Error : " << avg_err << std::endl;
	std::cout << "Min. Error : " << min_err << std::endl;
	std::cout << "Max. Error : " << max_err << std::endl;
	std::cout << std::endl;
}

void test_gradient_descent(){

	auto cos_f = [](double x){return cos(x);};
	auto cos_f_min = numeric::gradient_descent(cos_f, 2);
	std::cout << "test_gradient_descent" << std::endl;
	std::cout << "Min x : " << cos_f_min << ", Min y : " << cos_f(cos_f_min) << std::endl;
	std::cout << std::endl;
}

void test_linear_regression(){

	srand(time(NULL));
	auto c0 = rand() % 10 + 1.0;
	auto c1 = rand() % 10 + 1.0;
	auto f0 = [c0, c1](double x){ return c0 + c1 * x; };

	std::vector<double> xs;
	std::vector<double> ys;
	for(int i=0;i<20;i++){
		xs.push_back(i);

		// generate noisy y
		auto noise = rand() % 1000 / 1000;
		auto y = f0(i) + noise * 0.5;
		ys.push_back(y);
	}

	// linear regression
	auto coeffs = numeric::polynomial_regression(xs, ys, 2);

	// print
	std::cout << "test_linear_regression" << std::endl;
	std::cout << "Actual Coeffs. : {" << c0 << ", " << c1 << "}" << std::endl;
	std::cout << "Regres Coeffs. : {" << coeffs[0] << ", " << coeffs[1] << ", " << coeffs[2] << "}" << std::endl;

}

int main(){
	test_derivative();
	test_gradient_descent();
	test_linear_regression();
}
