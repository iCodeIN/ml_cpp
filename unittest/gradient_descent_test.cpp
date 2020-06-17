
#include "../gradient_descent.hpp"

#include <iostream>
#include <math.h>

/*
 * find min of cos(x)
 */
void test_gradient_descent_001()
{

    auto cos_f = [](float x)
    {
        return 0.0f + cos(x);
    };
    auto cos_f_min = numeric::gradient_descent(cos_f, 2);

    // print
    std::cout << std::endl;
    std::cout << "min y = cos(x)" << std::endl;
    std::cout << "Min x : " << cos_f_min << ", Min y : " << cos_f(cos_f_min) << std::endl;
    std::cout << std::endl;
}

/*
 * find min of sin(x)
 */
void test_gradient_descent_002()
{

    auto sin_f = [](float x)
    {
        return 0.0f + sin(x);
    };
    auto sin_f_min = numeric::gradient_descent(sin_f, 2);

    // print
    std::cout << std::endl;
    std::cout << "min y = sin(x)" << std::endl;
    std::cout << "Min x : " << sin_f_min << ", Min y : " << sin_f(sin_f_min) << std::endl;
    std::cout << std::endl;
}

/*
 * main
 */
int main()
{
    test_gradient_descent_001();
    test_gradient_descent_002();
}
