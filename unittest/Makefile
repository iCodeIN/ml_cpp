compile: 
	g++ -std=c++17 -o derivative derivative_test.cpp
	g++ -std=c++17 -o gradient_descent gradient_descent_test.cpp
	g++ -std=c++17 -o polynomial_regression polynomial_regression_test.cpp
	g++ -std=c++17 -o logistic_regression logistic_regression_test.cpp
	g++ -std=c++17 -o neural_network neural_network_test.cpp
	g++ -std=c++17 -o word2vec word2vec_test.cpp

test:
	./derivative
	./gradient_descent
	./polynomial_regression
	./logistic_regression
	./neural_network
	./word2vec

clean:
	astyle -q --style=allman --indent=spaces=4 --indent-classes --indent-switches --indent-cases --indent-namespaces --add-brackets --close-templates --suffix=none *.[ch]pp
	rm -f derivative
	rm -f gradient_descent
	rm -f polynomial_regression
	rm -f logistic_regression
	rm -f neural_network
	rm -f word2vec
