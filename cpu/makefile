
# Default target
all: unsafe

# Set compiler and additional flags according to the OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CC = clang
else
    CC = gcc
endif

naive:
	$(CC) -Iinclude src/*.c -o raytracer.exe -lm

optimize:
	$(CC) -Iinclude -O3 -flto src/*.c -o raytracer.exe -lm

unsafe:
	$(CC) -Iinclude -Ofast -march=native -flto src/*.c -o raytracer.exe -lm

debug:
	$(CC) -g -Iinclude -Ofast -march=native -flto src/*.c -o raytracer.exe -lm

clean:
	rm -f raytracer.exe
	rm -f render.bmp
