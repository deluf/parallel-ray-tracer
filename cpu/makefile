
# Default target
all: default

# Set compiler and additional flags according to the OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CC = clang
else
    CC = gcc
endif

default:
	$(CC) -Iinclude -O3 -ffast-math -march=native -flto src/*.c -o raytracer.exe -lm

clean:
	rm -f raytracer.exe
	rm -f render.bmp
