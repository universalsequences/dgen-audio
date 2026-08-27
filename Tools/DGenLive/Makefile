CC := clang
CPPFLAGS += -I../../toolchain/include -I../../Sources/DGenHostSupport/include
CFLAGS ?= -O2 -g
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic $(shell pkg-config --cflags alsa json-c)
LDLIBS += $(shell pkg-config --libs alsa json-c) -ldl -lpthread -lm

.PHONY: all clean test test-c test-elisp
all: dgen-live

dgen-live: dgen-live.c ../../Sources/DGenHostSupport/DGenHostSupport.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

tests/fixture.so: tests/fixture.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -shared -fPIC $< -lm -o $@

test: test-c test-elisp

test-c: dgen-live tests/fixture.so
	python3 tests/test_server.py

test-elisp:
	emacs --batch -Q -L . -l tests/dgenlisp-mode-tests.el -f ert-run-tests-batch-and-exit

clean:
	rm -f dgen-live tests/fixture.so
