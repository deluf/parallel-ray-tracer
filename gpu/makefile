SRC = $(wildcard src/*.cpp src/*.cu)

all:
	nvcc -O3 -use_fast_math -Iinclude -split-compile=0 -lineinfo -rdc=true -arch=sm_86 -dlto $(SRC) -o raytracer.exe