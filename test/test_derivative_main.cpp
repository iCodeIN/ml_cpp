#include "../derivative.hpp"

#include <iostream>
#include <math.h>

/*
 * derivative of sin
 */
void test_derivative_001(){
	
	// sin
	auto sin_f= [](double x){return sin(x);};

	// derivative should be cos
	auto cos_f = numeric::derivative(sin_f);

	// test series
	auto avg_err = 0.0;
	auto max_err = 0.0;
	auto min_err = 1.0;
	for(int i=0;i<360;i++){
		auto j = i * 0.01745329;
		auto k = abs(cos_f(j) - cos(j));
		avg_err += k;
		min_err = k < min_err ? k : min_err;
		max_err = k > max_err ? k : max_err;
	}
	avg_err /= 360;

	// print
	std::cout << std::endl;
	std::cout << "derivative of sin(x) == cos(x)" << std::endl;
	std::cout << "avg error : " << avg_err << std::endl;
	std::cout << "min error : " << min_err << std::endl;
	std::cout << "max error : " << max_err << std::endl;
}

/*
 * derivative of cos
 */
void test_derivative_002(){

	// cos
	auto cos_f= [](double x){return cos(x);};

	// derivative should be -sin
	auto sin_f = numeric::derivative(cos_f);

	// test series
	auto avg_err = 0.0;
	auto max_err = 0.0;
	auto min_err = 1.0;
	for(int i=0;i<360;i++){
		auto j = i * 0.01745329;
		auto k = abs(sin_f(j) + sin(j));
		avg_err += k;
		min_err = k < min_err ? k : min_err;
		max_err = k > max_err ? k : max_err;
	}
	avg_err /= 360;

	// print
	std::cout << std::endl;
	std::cout << "derivative of cos(x) == -sin(x)" << std::endl;
	std::cout << "avg error : " << avg_err << std::endl;
	std::cout << "min error : " << min_err << std::endl;
	std::cout << "max error : " << max_err << std::endl;

}

/*
 * derivative of tan
 */
void test_derivative_003(){

	// cos
	auto tan_f= [](double x){return tan(x);};

	// derivative should be -sin
	auto sec_f = numeric::derivative(tan_f);

	// test series
	auto avg_err = 0.0;
	auto max_err = 0.0;
	auto min_err = 1.0;
	for(int i=0;i<360;i++){
		// asymptotes
		if(i == 90 || i == 270)
			continue;
		auto j = i * 0.01745329;
		auto sec = 1.0 / ( cos(j) * cos(j) );
		auto k = abs(sec_f(j) - sec);
		avg_err += k;
		min_err = k < min_err ? k : min_err;
		max_err = k > max_err ? k : max_err;
	}
	avg_err /= 360;

	// print
	std::cout << std::endl;
	std::cout << "derivative of tan(x) == sec(x)**2" << std::endl;
	std::cout << "avg error : " << avg_err << std::endl;
	std::cout << "min error : " << min_err << std::endl;
	std::cout << "max error : " << max_err << std::endl;

}

/*
 * main
 */
int main(){
	test_derivative_001();
	test_derivative_002();
	test_derivative_003();
}
