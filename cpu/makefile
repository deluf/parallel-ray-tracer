
# Default target
all: optimize

# Set compiler and additional flags according to the OS
ifeq ($(OS),Windows_NT)
    CC = gcc
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        CC = clang
    else
        $(error Unsupported OS)
    endif
endif

naive:
	$(CC) -Iinclude src/*.c -o raytracer.exe

optimize:
	$(CC) -Iinclude -O3 -flto src/*.c -o raytracer.exe

unsafe:
	$(CC) -Iinclude -Ofast -flto -march=native -funroll-loops -fprefetch-loop-arrays -ftree-vectorize -fopenmp src/*.c -o raytracer.exe

clean:
	rm -f raytracer.exe
	rm -f render.bmp