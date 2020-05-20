#include "../polynomial_regression.hpp"

#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <time.h>

void test_polynomial_regression(double max_noise, int degree){

	srand(time(NULL));
	std::vector<double> coeffs;
	for(int i=0;i<=degree;i++){
		auto c = rand() % 10 + 1.0;
		coeffs.push_back(c);
	}
	auto f0 = [coeffs](double x){ 
		auto y = 0.0;
		for(int i=0;i<coeffs.size();i++){
			y += pow(x, i) * coeffs[i];
		}
		return y;
	};
	
	// print
	std::cout << std::endl;
	std::cout << "test_linear_regression" << std::endl;

	// build xs/ys
	std::vector<double> xs;
	std::vector<double> ys;
	for(int i=0;i<20;i++){
		xs.push_back(i);

		// generate noisy y
		auto noise = (rand() % 1000) / 1000.0;
		auto y = f0(i) + noise * max_noise;
		std::cout << i << "\t" << y << std::endl;
		ys.push_back(y);
	}

	// linear regression
	auto approx_coeffs = numeric::polynomial_regression(xs, ys, degree);

	// print
	std::cout << "Actual Coeffs. : {";
	for(int i=0;i<coeffs.size();i++){
		std::cout << coeffs[i] << ", ";
	}
	std::cout << "}" << std::endl;

	std::cout << "Regres Coeffs. : {";
	for(int i=0;i<approx_coeffs.size();i++){
		std::cout << approx_coeffs[i] << ", ";
	}
	std::cout << "}" << std::endl;
}

int main(){

	test_polynomial_regression(0.5, 1);
	test_polynomial_regression(0.5, 1);
	test_polynomial_regression(0.5, 1);

	test_polynomial_regression(0.5, 2);
	test_polynomial_regression(0.5, 2);

	test_polynomial_regression(0.5, 3);
	test_polynomial_regression(0.5, 3);
}
