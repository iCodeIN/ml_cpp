#include "../gradient_descent.hpp"
#include "../matrix.hpp"
#include "../neural_network.hpp"

#include <assert.h>
#include <iostream>
#include <vector>
#include <string>

/*
 * IO
 */

const std::vector<std::string> explode(const std::string& s, const char& c)
{
	std::string buff{""};
	std::vector<std::string> v;
	for(auto n:s)
	{
		if(n != c) buff+=n; else
		if(n == c && buff != "") { v.push_back(buff); buff = ""; }
	}
	if(buff != "") v.push_back(buff);
	return v;
}

void read_network(){
}

void store_network(){
}

/*
 * TRAINING
 */

void train_network(){
}

/*
 * FORWARD
 */

void use_network(){
}

/*
 * MAIN METHOD
 */

bool has_arg(int argc, char* argv[], std::string key){
	for(int i=0;i<argc;i++){
		if(key.compare(argv[i]) == 0)
			return true;
	}
	return false;
}

std::string arg(int argc, char* argv[], std::string key){
	for(int i=0;i<argc;i++){
		if(key.compare(argv[i])==0)
			return argv[i+1];
	}
}

int main(int argc, char* argv[]){

	// read input data from cin
	std::vector<std::vector<float>> xs;
	std::vector<std::vector<float>> ys;
	for(std::string line; std::getline(std::cin, line);){
		auto tokens = explode(line, '\t');

		// xs
		std::vector<float> row_xs;
		for(int i=0;i<tokens.size() - 1;i++){
			row_xs.push_back(std::stof(tokens[i]));
		}
		assert(xs.size() == 0 || xs[xs.size() - 1].size() == row_xs.size());
		xs.push_back(row_xs);

		// ys
		std::vector<float> row_ys;
		row_ys.push_back(std::stof(tokens[tokens.size() - 1]));
		assert(ys.size() == 0 ||  ys[ys.size() - 1].size() == row_ys.size());
		ys.push_back(row_ys);
	}

	// determine network topology
	auto nn = nn::init_neural_network({1, 1});
	if(has_arg(argc, argv, "-size")){
		auto tokens = explode(arg(argc, argv, "-size"), ',');
		std::vector<int> dims;
		for(int i=0;i<tokens.size();i++){
			dims.push_back(std::stoi(tokens[i]));
		}
		nn = nn::init_neural_network(dims);
	}
	if(has_arg(argc, argv, "-i")){
	}
	assert(nn.size() > 0);
	assert(matrix::rows(nn[0]) == matrix::cols(xs));

	// determine number of iterations
	auto iterations = 1024;
	if(has_arg(argc, argv, "-iterations")){
		iterations = std::stoi(arg(argc, argv, "-iterations"));
		assert(iterations > 0);
	}

	// train
	nn = nn::train(xs, ys, nn, numeric::constant_learning_rate(0.1), iterations);

	// store
	if(has_arg(argc, argv, "-o")){
		
	}

}


