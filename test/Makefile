test: 
	g++ -std=c++17 -o test_derivative test_derivative_main.cpp
	g++ -std=c++17 -o test_gradient_descent test_gradient_descent_main.cpp
	g++ -std=c++17 -o test_polynomial_regression test_polynomial_regression_main.cpp
	g++ -std=c++17 -o test_logistic_regression test_logistic_regression_main.cpp
	g++ -std=c++17 -o test_neural_network test_neural_network_main.cpp
	./test_derivative
	./test_gradient_descent
	./test_polynomial_regression
	./test_logistic_regression
	./test_neural_network

clean:
	astyle -q --style=allman --indent=spaces=4 --indent-classes --indent-switches --indent-cases --indent-namespaces --add-brackets --close-templates --suffix=none *.cpp
	rm -f test_derivative
	rm -f test_gradient_descent
	rm -f test_polynomial_regression
	rm -f test_logistic_regression
	rm -f test_neural_network
