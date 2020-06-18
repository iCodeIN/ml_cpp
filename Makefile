clean:
	astyle -q --style=allman --indent=spaces=4 --indent-classes --indent-switches --indent-cases --indent-namespaces --add-brackets --close-templates --suffix=none *.[ch]pp
	cd unittest && $(MAKE) clean
	cd executable && $(MAKE) clean

compile:
	cd unittest && $(MAKE) compile
	cd executable && $(MAKE) compile

test:
	cd unittest && $(MAKE) test

