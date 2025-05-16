# Default target
all: default

# Set compiler and additional flags according to the OS
SM_ARCH = sm_86
SRC = $(wildcard src/*.cpp src/*.cu)

default:
	nvcc -O3 -use_fast_math -Iinclude -split-compile=0 -g -lineinfo -rdc=true -dlto $(SRC) -o raytracer.exe -arch=$(SM_ARCH)

clean:
	rm -f raytracer.exe
	rm -f render.bmp
