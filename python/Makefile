
all: ext3

allchecks: _tskitmodule.c
#	CFLAGS="-std=c99 --coverage -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \

	CFLAGS="-std=c99 -Wall -Wextra -Werror -Wno-unused-parameter -Wno-cast-function-type" \
	bear -- python3 setup.py build_ext --debug --inplace -j8

ext3: _tskitmodule.c
	bear -- python3 setup.py build_ext --inplace

ctags:
	ctags lib/*.c lib/*.h tskit/*.py

clean:
	rm -f *.so *.o tags
	rm -fR build
