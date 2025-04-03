
# Default target
all: build

# Set compiler and additional flags according to the OS
ifeq ($(OS),Windows_NT)
    CC = gcc
    CFLAGS = -mavx
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        CC = clang
        CFLAGS = 
    else
        $(error Unsupported OS)
    endif
endif

build:
	$(CC) $(CFLAGS) -Iinclude -O3 -flto src/*.c -o raytracer.exe

clean:
	rm -f raytracer.exe
	rm -f img.bmp