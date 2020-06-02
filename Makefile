clean:
	astyle -q --style=allman --indent=spaces=4 --indent-classes --indent-switches --indent-cases --indent-namespaces --add-brackets --close-templates --suffix=none *.hpp
	cd test && $(MAKE) clean

test:
	cd test && $(MAKE) test

