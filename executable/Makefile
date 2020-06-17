compile:
	g++ -std=c++17 -o nn nn_main.cpp

clean:
	astyle -q --style=allman --indent=spaces=4 --indent-classes --indent-switches --indent-cases --indent-namespaces --add-brackets --close-templates --suffix=none *.cpp
	astyle -q --style=allman --indent=spaces=4 --indent-classes --indent-switches --indent-cases --indent-namespaces --add-brackets --close-templates --suffix=none *.hpp
	rm -f nn
